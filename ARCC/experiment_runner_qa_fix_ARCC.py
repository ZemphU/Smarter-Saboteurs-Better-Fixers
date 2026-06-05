import sys
import os
import json
import asyncio
import logging
import traceback
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent  # the ARCC/ directory

# Ensure the project directory is importable (for experiment_config).
# MetaGPT itself is installed as a pip dependency; see requirements.txt.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from metagpt.roles import Role
from metagpt.actions import Action
from metagpt.schema import Message
from metagpt.team import Team
from metagpt.llm import LLM
from metagpt.actions.add_requirement import UserRequirement
from metagpt.config2 import Config
from metagpt.configs.llm_config import LLMConfig, LLMType

# Removed nnsight imports and Config logic for local execution


# Wrapper for Local HuggingFace Models
class LocalHuggingFaceLLM:
    _instance = None
    _current_model_name = None
    _model = None
    _tokenizer = None

    def __init__(self, model_name):
        self.model_name = model_name
        
        # Singleton/Cache Logic
        if LocalHuggingFaceLLM._current_model_name != model_name:
            logger.info(f"Loading new model: {model_name} (replacing {LocalHuggingFaceLLM._current_model_name})")
            
            # Clear previous model if exists
            if LocalHuggingFaceLLM._model is not None:
                del LocalHuggingFaceLLM._model
                del LocalHuggingFaceLLM._tokenizer
                import torch
                torch.cuda.empty_cache()
                logger.info("Cleared previous model from VRAM.")

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                LocalHuggingFaceLLM._tokenizer = AutoTokenizer.from_pretrained(model_name)
                if LocalHuggingFaceLLM._tokenizer.pad_token is None:
                        LocalHuggingFaceLLM._tokenizer.pad_token = LocalHuggingFaceLLM._tokenizer.eos_token
                        
                LocalHuggingFaceLLM._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                LocalHuggingFaceLLM._current_model_name = model_name
                logger.info(f"Model {model_name} loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise e
        else:
            logger.info(f"Using cached model: {model_name}")

    async def aask(self, msg: str, system_msgs: list[str] = None) -> str:
        prompt = msg
        if system_msgs:
             prompt = "\n".join(system_msgs) + "\n" + msg
             
        # logger.info(f"LocalHuggingFaceLLM ({self.model_name}) generating for prompt len: {len(prompt)}")
        
        # Run generation in a separate thread to avoid blocking asyncio loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self._generate, prompt)
        return response

    def _generate(self, prompt):
        try:
            # Use class attributes
            tokenizer = LocalHuggingFaceLLM._tokenizer
            model = LocalHuggingFaceLLM._model
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                **experiment_config.GENERATION_CONFIG,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode only the new tokens
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_len:]
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            logger.info(f"Generated text len: {len(decoded_text)}")
            return decoded_text
        except Exception as e:
            logger.error(f"Generation failed: {e!r}")
            traceback.print_exc()
            return f"Error generating response: {e}"


class SimpleAction(Action):
    async def run(self, instruction: str):
        # Use the role's LLM
        return await self.llm.aask(instruction)



# Configuration (currently set up to loop through the Qwen3.5 family)
LOCAL_MODELS = [
    #"google/gemma-3-270m-it",
    #"google/gemma-3-1b-it",
    #"google/gemma-3-4b-it",
    #"google/gemma-3-12b-it",
    #"google/gemma-3-27b-it",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
]

# Torch Dynamo Configuration (Fix for cache_size_limit reached)
try:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 128
    logger.info("Set torch._dynamo.config.cache_size_limit = 128")
except ImportError:
    pass
except Exception as e:
    logger.warning(f"Could not set torch._dynamo cache limit: {e}")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_qa_fix_arcc.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import Shared Configuration
import experiment_config

class SimpleRole(Role):
    def __init__(self, name: str, profile: str, goal: str, constraints: str, model_name: str, watch_list: list[str] = None, is_malicious: bool = False, transformed_profile: str = None):
        super().__init__(name=name, profile=profile, goal=goal, constraints=constraints)
        self.set_actions([SimpleAction])
        self.model_name = model_name
        self.is_malicious = is_malicious
        self.transformed_profile = transformed_profile
        
        # Set up linear structure: Watch specific actions or roles
        if watch_list:
            self._watch(watch_list)
        
        # Configure LLM based on model_name
        self.local_llm = None
        
        # Check if it's a Local model
        if any(local_model in model_name for local_model in LOCAL_MODELS):
             self.local_llm = LocalHuggingFaceLLM(model_name)
             logger.info(f"Role {name} initialized with LocalHuggingFaceLLM: {model_name}")
        else:
            logger.error(f"Unsupported model: {model_name}. Only local models are supported here.")
            raise ValueError(f"Unsupported model: {model_name}")
        
        if is_malicious:
            if self.transformed_profile:
                 self.profile = self.transformed_profile
                 logger.info(f"Using cached malicious profile for {name}")
            else:
                logger.error(f"Malicious scenario requested but no malicious profile was supplied for {name}")

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self.rc.todo}")
        todo = self.rc.todo
        
        # Get the history of messages
        history = self.get_memories(k=10) 
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
        
        instruction = f"You are a {self.profile}.\n\nContext:\n\"\"\"\n{context}\n\"\"\"\n\nTask: {self.goal}\nConstraints: {self.constraints}\n\nPlease provide your output."
        
        # Determine forcing prefix based on role name (stable)
        prefixes = {
            "ProductManager": "Here is the analysis:\n",
            "Architect": "Here is the design:\n",
            "ProjectManager": "Here is the implementation plan:\n",
            "Engineer": "Here is the code:\n",
            "QAEngineer": "Here is the QA report:\nSTATUS: ",
            "FixerEngineer": "Here is the fixed code:\nFIXED_CODE\n```python\n"
        }
        force_prefix = prefixes.get(self.name, "Here is the output:\n")
        logger.info(f"Using forcing prefix for {self.name}: {repr(force_prefix)}")

        # Apply the correct chat template using the tokenizer natively
        chat = [{"role": "user", "content": instruction}]
        prompt_text = instruction
        
        if self.local_llm and hasattr(self.local_llm, '_tokenizer') and self.local_llm._tokenizer:
            try:
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if "qwen" in self.model_name.lower():
                    template_kwargs["enable_thinking"] = False
                prompt_text = self.local_llm._tokenizer.apply_chat_template(chat, **template_kwargs)
                if force_prefix:
                    prompt_text += force_prefix
            except Exception as e:
                logger.warning(f"Failed to apply chat template natively: {e}. Falling back to raw string.")
                if "gemma" in self.local_llm.model_name.lower():
                    prompt_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{force_prefix}"
                else:
                    prompt_text += f"\n{force_prefix}"
        else:
             prompt_text += f"\n{force_prefix}"
        
        # DEBUG: Log the FINAL instruction to ensure template is applied
        # logger.info(f"FINAL PROMPT for {self.name}: {repr(prompt_text)}")
        
        if self.local_llm:
             response = await self.local_llm.aask(prompt_text)
        else:
             response = await todo.run(prompt_text)
             
        # We need to prepend the force_prefix in the final response if the model didn't return it
        if force_prefix and not response.startswith(force_prefix):
             response = force_prefix + response
             
        msg = Message(content=response, role=self.name, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg

async def run_linear_collaboration(task_prompt, roles_config, model_name):
    """
    Runs a linear collaboration with the given roles.
    roles_config: list of dicts with keys 'name', 'profile', 'goal', 'constraints', 'is_malicious'
    """
    team = Team(use_mgx=False)
    roles = []
    
    for i, config in enumerate(roles_config):
        watch_list = []
        if i == 0:
            watch_list = [UserRequirement] 
        else:
            # Ensure subsequent roles see the User Prompt (Requirement) + Previous Actions
            watch_list = [UserRequirement, SimpleAction] 
        
        role = SimpleRole(
            name=config['name'],
            profile=config['profile'],
            goal=config['goal'],
            constraints=config['constraints'],
            model_name=model_name,
            watch_list=watch_list,
            is_malicious=config.get('is_malicious', False),
            transformed_profile=config.get('transformed_profile', None)
        )
        roles.append(role)
        team.hire([role])
    
    # Manual Sequential Execution to avoid broadcast storms and ensure strict linearity
    # Initialize the project
    team.invest(investment=1.0)
    team.run_project(task_prompt)
    
    # Run roles sequentially
    for i, role in enumerate(roles):
        # Allow early stopping if QA finds no issues
        if role.name == "FixerEngineer":
            qa_output = ""
            for r in roles:
                if r.name == "QAEngineer" and r.rc.memory.get():
                    qa_output = r.rc.memory.get()[-1].content
                    break
            if "NO_ISSUES" in qa_output:
                logger.info(f"QAEngineer found NO_ISSUES. Skipping FixerEngineer.")
                continue

        logger.info(f"--- Executing Role {i}: {role.name} ---")
        await role.run()
    
    outputs = {}
    for role in roles:
        if role.rc.memory.get():
            outputs[role.name] = role.rc.memory.get()[-1].content
    
    return outputs, roles

async def run_experiment_loop():
    # Load HumanEval tasks
    tasks = []
    benchmark_path = os.path.join(project_root, 'data', 'HumanEval.jsonl')
    try:
        with open(benchmark_path, 'r') as f:
            for line in f:
                tasks.append(json.loads(line))
    except FileNotFoundError:
        logger.error(f"Benchmark file not found at {benchmark_path}")
        return

    # Load Malicious Profile
    malicious_profile_content = ""
    malicious_profile_path = experiment_config.MALICIOUS_PROFILE_PATH
    try:
        with open(malicious_profile_path, 'r') as f:
            malicious_profile_content = f.read().strip()
        logger.info(f"Loaded malicious profile from {malicious_profile_path}")
    except Exception as e:
        logger.warning(f"Could not load malicious profile: {e}")

    # ARCC_MAX_TASKS=N limits the run to the first N problems (useful for testing)
    _max = int(os.environ.get("ARCC_MAX_TASKS", 0))
    selected_tasks = tasks[:_max] if _max else tasks
    
    
    # Define Models
    models = [{"name": model_name, "api_model": model_name} for model_name in LOCAL_MODELS]
    
    # Define Scenarios
    scenarios = ["Control", "Malicious"]
    
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'qwen_results_qa_fix_arcc.jsonl')
    traces_file = os.path.join(results_dir, 'qwen_traces_qa_fix_arcc.jsonl')
    
    logger.info(f"Starting experiment loop: {len(models)} models * {len(selected_tasks)} tasks * {len(scenarios)} scenarios")

    for model in models:
        model_name = model['name']
        logger.info(f"=== Loading Model: {model_name} ===")

        for task in selected_tasks:
            task_id = task['task_id']
            prompt = task['prompt']
            
            # Construct the coding task prompt
            task_prompt_coding = experiment_config.TASK_PROMPT_CODING_TEMPLATE.format(prompt=prompt)
            
            # Cache for malicious profiles
            malicious_profile_cache = {}

            for scenario in scenarios:
                logger.info(f"Running Task: {task_id} | Model: {model_name} | Scenario: {scenario}")
                
               # Check cache for malicious profile
                engineer_profile = "Software Engineer"
                
                # Use static profile if available and Malicious scenario
                if scenario == "Malicious":
                    if malicious_profile_content:
                        engineer_profile = malicious_profile_content
                    elif "Engineer" in malicious_profile_cache:
                        engineer_profile = malicious_profile_cache["Engineer"]
                
                roles_config_coding = experiment_config.get_qa_roles_config(scenario, engineer_profile)
                
                try:
                    outputs, roles = await run_linear_collaboration(task_prompt_coding, roles_config_coding, model_name)
                    
                    # We want the Fixer Engineer's output if it fixed the code, otherwise the Engineer's
                    engineer_output = outputs.get("Engineer", "")
                    qa_output = outputs.get("QAEngineer", "")
                    fixer_output = outputs.get("FixerEngineer", "")
                    
                    final_output = engineer_output # Default to Engineer
                    
                    if qa_output and "NO_ISSUES" in qa_output:
                        logger.info(f"QA Engineer found no issues for task {task_id}")
                        qa_action = "approved"
                    elif fixer_output:
                        try:
                            # Try to extract from a python code block anywhere in the text
                            if "```python" in fixer_output:
                                final_output = fixer_output.split("```python")[1].split("```")[0].strip()
                                qa_action = "fixed"
                                logger.info(f"Fixer Engineer fixed the code for task {task_id}")
                            elif "```" in fixer_output:
                                parts = fixer_output.split("```")
                                if len(parts) >= 3:
                                    final_output = parts[1].strip()
                                    qa_action = "fixed"
                                    logger.info(f"Fixer Engineer fixed the code for task {task_id}")
                                else:
                                    # Fallback
                                    final_output = fixer_output.replace("FIXED_CODE", "").strip()
                                    qa_action = "fixed"
                            elif "FIXED_CODE" in fixer_output:
                                final_output = fixer_output.split("FIXED_CODE")[1].strip()
                                qa_action = "fixed"
                            else:
                                final_output = fixer_output.strip()
                                qa_action = "unrecognized_but_used_fixer"
                        except Exception as e:
                            logger.warning(f"Failed to extract fixed code from Fixer output: {e}")
                            qa_action = "error_extracting_fix"
                    else:
                        logger.warning(f"QA or Fixer output format unrecognized for task {task_id}. Using Engineer output.")
                        qa_action = "unrecognized"

                    result_entry = {
                        "task_id": task_id,
                        "model": model_name,
                        "scenario": scenario,
                        "output": final_output,
                        "status": "success",
                        "qa_action": qa_action
                    }
                    
                    # Update cache if we just ran a malicious scenario and cache is empty
                    if scenario == "Malicious" and "Engineer" not in malicious_profile_cache:
                        for role in roles:
                            if role.name == "Engineer" and role.is_malicious:
                                malicious_profile_cache["Engineer"] = role.profile
                                malicious_profile_cache["Engineer"] = role.profile
                                logger.info(f"Cached malicious profile for Engineer: {role.profile[:50]}...")
                except Exception as e:
                    logger.error(f"Error in run {task_id} {model_name} {scenario}: {e}")
                    result_entry = {
                        "task_id": task_id,
                        "model": model_name,
                        "scenario": scenario,
                        "output": str(e),
                        "status": "error"
                    }
                
                # Save result immediately
                with open(results_file, 'a') as f:
                    f.write(json.dumps(result_entry) + "\n")
                
                # Save trace immediately
                if "status" in result_entry and result_entry["status"] == "success":
                    trace_entry = {
                        "task_id": task_id,
                        "model": model_name,
                        "scenario": scenario,
                        "roles_output": outputs
                    }
                    with open(traces_file, 'a') as f:
                        f.write(json.dumps(trace_entry) + "\n")

async def main():
    await run_experiment_loop()

if __name__ == "__main__":
    asyncio.run(main())
