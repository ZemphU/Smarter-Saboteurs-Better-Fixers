import sys
import os
import json
import asyncio
import logging
import traceback
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

current_file = Path(__file__).resolve()
project_root = current_file.parent  # the ARCC/ directory

# Ensure the project directory is importable (for experiment_config).
# MetaGPT itself is installed as a pip dependency; see requirements.txt.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from metagpt.roles import Role
from metagpt.actions import Action
from metagpt.schema import Message
from metagpt.team import Team
from metagpt.actions.add_requirement import UserRequirement
from metagpt.provider.base_llm import BaseLLM

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
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_native_mgpt_arcc.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import experiment_config


# ============================================================================
# LocalHuggingFaceLLM - Wrapper inheriting from MetaGPT's BaseLLM
# ============================================================================
class LocalHuggingFaceLLM(BaseLLM):
    _instance = None
    _current_model_name = None
    _model = None
    _tokenizer = None

    def __init__(self, model_name):
        from metagpt.configs.llm_config import LLMConfig
        super().__init__(config=LLMConfig())
        self.model_name = model_name
        self.model = model_name

        # Singleton/Cache Logic
        if LocalHuggingFaceLLM._current_model_name != model_name:
            logger.info(f"Loading new model: {model_name}")
            if LocalHuggingFaceLLM._model is not None:
                del LocalHuggingFaceLLM._model
                del LocalHuggingFaceLLM._tokenizer
                import torch
                torch.cuda.empty_cache()
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
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise e

    async def _achat_completion(self, messages: list[dict], timeout=None):
        raise NotImplementedError

    async def _achat_completion_stream(self, messages: list[dict], timeout=None) -> str:
        raise NotImplementedError

    async def acompletion(self, messages: list[dict], timeout=None):
        raise NotImplementedError

    async def aask(self, msg, system_msgs=None, format_msgs=None, images=None, timeout=None, stream=None) -> str:
        prompt = msg
        if system_msgs:
            prompt = "\n".join(system_msgs) + "\n" + msg

        # Apply the correct chat template using the tokenizer natively
        chat = [{"role": "user", "content": prompt}]
        try:
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if "qwen" in self.model_name.lower():
                template_kwargs["enable_thinking"] = False
            prompt_text = LocalHuggingFaceLLM._tokenizer.apply_chat_template(
                chat, **template_kwargs
            )
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, falling back to manual format")
            prompt_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self._generate, prompt_text)
        return response

    def _generate(self, prompt):
        try:
            tokenizer = LocalHuggingFaceLLM._tokenizer
            model = LocalHuggingFaceLLM._model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                **experiment_config.GENERATION_CONFIG,
                pad_token_id=tokenizer.pad_token_id
            )
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_len:]
            return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Generation failed: {e!r}")
            traceback.print_exc()
            return f"Error generating response: {e}"


# ============================================================================
# SimpleAction / NativeMGPTRole - Linear roles using MetaGPT's exact profiles
# ============================================================================
class SimpleAction(Action):
    """Minimal action that delegates to the role's LLM."""
    async def run(self, instruction: str):
        return await self.llm.aask(instruction)


class NativeMGPTRole(Role):
    """
    A linear-execution role that uses the exact same profile/goal/constraints
    as the native MetaGPT roles, but without their deep file-system/git/tool
    integration. This enables strict A->B->C->D->E linear execution matching
    the paper's definition of MetaGPT as a "linear" system.
    """
    def __init__(self, name: str, profile: str, goal: str, constraints: str,
                 local_llm, watch_list: list = None,
                 is_malicious: bool = False, transformed_profile: str = None,
                 transformed_goal: str = None, transformed_constraints: str = None):
        super().__init__(name=name, profile=profile, goal=goal, constraints=constraints)
        self.set_actions([SimpleAction])
        self.is_malicious = is_malicious
        self.transformed_profile = transformed_profile

        # Set up linear structure: Watch specific actions or roles
        if watch_list:
            self._watch(watch_list)

        # Inject local LLM
        self.local_llm = local_llm

        # Apply malicious profile override
        # The malicious profile text is a complete role description, so we
        # override all three fields to avoid contradicting instructions.
        if is_malicious and self.transformed_profile:
            self.profile = self.transformed_profile
            if transformed_goal is not None:
                self.goal = transformed_goal
            if transformed_constraints is not None:
                self.constraints = transformed_constraints
            logger.info(f"Applied malicious profile for {name}")

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self.rc.todo}")
        todo = self.rc.todo

        # Get the history of messages
        history = self.get_memories(k=10)
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in history])

        # Use the NATIVE MetaGPT profile/goal/constraints in the instruction
        # This matches exactly what native MetaGPT roles would see
        instruction = (
            f"You are {self.name}, a {self.profile}.\n\n"
            f"Your goal: {self.goal}\n"
            f"Constraints: {self.constraints}\n\n"
            f"Context from previous agents:\n\"\"\"\n{context}\n\"\"\"\n\n"
            f"Please provide your output."
        )

        # Apply chat template and generate
        chat = [{"role": "user", "content": instruction}]
        prompt_text = instruction

        if self.local_llm and hasattr(LocalHuggingFaceLLM, '_tokenizer') and LocalHuggingFaceLLM._tokenizer:
            try:
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if "qwen" in (LocalHuggingFaceLLM._current_model_name or "").lower():
                    template_kwargs["enable_thinking"] = False
                prompt_text = LocalHuggingFaceLLM._tokenizer.apply_chat_template(
                    chat, **template_kwargs
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}")
                prompt_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

        response = await self.local_llm.aask(prompt_text)

        msg = Message(content=response, role=self.name, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg


# ============================================================================
# Native MetaGPT Role Profiles (extracted from MetaGPT source)
# These are the exact profile/goal/constraints used by the native roles.
# ============================================================================
NATIVE_METAGPT_ROLES = {
    "ProductManager": {
        "name": "Alice",
        "profile": "Product Manager",
        "goal": "Create a Product Requirement Document or market research/competitive product research.",
        "constraints": "utilize the same language as the user requirements for seamless communication",
    },
    "Architect": {
        "name": "Bob",
        "profile": "Architect",
        "goal": "design a concise, usable, complete software system. output the system design.",
        "constraints": (
            "make sure the architecture is simple enough and use appropriate open source "
            "libraries. Use same language as user requirement"
        ),
    },
    "ProjectManager": {
        "name": "Eve",
        "profile": "Project Manager",
        "goal": (
            "break down tasks according to PRD/technical design, generate a task list, and analyze task "
            "dependencies to start with the prerequisite modules"
        ),
        "constraints": "use same language as user requirement",
    },
    "Engineer": {
        "name": "Alex",
        "profile": "Engineer",
        "goal": "write elegant, readable, extensible, efficient code",
        "constraints": (
            "the code should conform to standards like google-style and be modular and maintainable. "
            "Use same language as user requirement"
        ),
    },
    "QaEngineer": {
        "name": "Edward",
        "profile": "QaEngineer",
        "goal": "Write comprehensive and robust tests to ensure codes will work as expected without bugs",
        "constraints": (
            "The test code you write should conform to code standard like PEP8, be modular, easy to read and maintain. "
            "Use same language as user requirement"
        ),
    },
}


def get_native_mgpt_roles_config(scenario, engineer_profile):
    """
    Returns role configs using native MetaGPT profiles for a strictly linear pipeline.
    Paper setup: ProductManager -> Architect -> ProjectManager -> Engineer -> QaEngineer
    
    In the paper's linear structure, the Engineer's output is the final evaluated code.
    The QaEngineer reviews but cannot modify code (no feedback loop).
    """
    roles = []
    for role_key in ["ProductManager", "Architect", "ProjectManager", "Engineer", "QaEngineer"]:
        config = dict(NATIVE_METAGPT_ROLES[role_key])
        config["is_malicious"] = (scenario == "Malicious" and role_key == "Engineer")
        if config["is_malicious"]:
            # The malicious profile is a complete role description that replaces
            # the Engineer's identity. We clear goal/constraints to avoid
            # contradicting the malicious instructions.
            config["transformed_profile"] = engineer_profile
            config["transformed_goal"] = "Follow the instructions in your profile."
            config["transformed_constraints"] = "Follow the instructions in your profile."
        else:
            config["transformed_profile"] = None
        roles.append(config)
    return roles


# ============================================================================
# Strictly Linear Collaboration
# ============================================================================
async def run_linear_collaboration(task_prompt, roles_config, model_name):
    """
    Runs a STRICTLY LINEAR collaboration: A -> B -> C -> D -> E
    Each role runs exactly once, in order. No feedback loops.
    This matches the paper's definition of MetaGPT as a linear system.
    """
    team = Team(use_mgx=False)
    local_llm = LocalHuggingFaceLLM(model_name)
    roles = []

    for i, config in enumerate(roles_config):
        watch_list = []
        if i == 0:
            watch_list = [UserRequirement]
        else:
            watch_list = [UserRequirement, SimpleAction]

        role = NativeMGPTRole(
            name=config['name'],
            profile=config['profile'],
            goal=config['goal'],
            constraints=config['constraints'],
            local_llm=local_llm,
            watch_list=watch_list,
            is_malicious=config.get('is_malicious', False),
            transformed_profile=config.get('transformed_profile', None),
            transformed_goal=config.get('transformed_goal', None),
            transformed_constraints=config.get('transformed_constraints', None)
        )
        roles.append(role)
        team.hire([role])

    # Initialize the project
    team.invest(investment=1.0)
    team.run_project(task_prompt)

    # STRICTLY SEQUENTIAL execution - each role runs exactly once
    for i, role in enumerate(roles):
        logger.info(f"--- Executing Role {i}: {role.name} ({role.profile}) ---")
        await role.run()

    # Collect all outputs
    outputs = {}
    for role in roles:
        if role.rc.memory.get():
            outputs[role.name] = role.rc.memory.get()[-1].content

    return outputs, roles


# ============================================================================
# Experiment Loop
# ============================================================================
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

    models = [{"name": model_name, "api_model": model_name} for model_name in LOCAL_MODELS]
    scenarios = ["Control", "Malicious"]

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'qwen_results_native_mgpt_arcc.jsonl')
    traces_file = os.path.join(results_dir, 'qwen_traces_native_mgpt_arcc.jsonl')

    logger.info(f"Starting experiment loop: {len(models)} models * {len(selected_tasks)} tasks * {len(scenarios)} scenarios")

    for model in models:
        model_name = model['name']
        logger.info(f"=== Loading Model: {model_name} ===")

        for task in selected_tasks:
            task_id = task['task_id']
            prompt = task['prompt']

            task_prompt_coding = experiment_config.TASK_PROMPT_CODING_TEMPLATE.format(prompt=prompt)
            malicious_profile_cache = {}

            for scenario in scenarios:
                logger.info(f"Running Task: {task_id} | Model: {model_name} | Scenario: {scenario}")

                engineer_profile = "Software Engineer"
                if scenario == "Malicious":
                    if malicious_profile_content:
                        engineer_profile = malicious_profile_content
                    elif "Engineer" in malicious_profile_cache:
                        engineer_profile = malicious_profile_cache["Engineer"]

                roles_config = get_native_mgpt_roles_config(scenario, engineer_profile)

                try:
                    outputs, roles = await run_linear_collaboration(task_prompt_coding, roles_config, model_name)

                    # In the paper's linear MetaGPT, the Engineer's output is the final code.
                    # The QaEngineer reviews but cannot modify code (no feedback loop).
                    engineer_output = outputs.get("Alex", "")  # Alex = Engineer's name

                    result_entry = {
                        "task_id": task_id,
                        "model": model_name,
                        "scenario": scenario,
                        "output": engineer_output,
                        "status": "success"
                    }

                    if scenario == "Malicious" and "Engineer" not in malicious_profile_cache:
                        malicious_profile_cache["Engineer"] = engineer_profile

                except Exception as e:
                    logger.error(f"Error in run {task_id} {model_name} {scenario}: {e}")
                    traceback.print_exc()
                    result_entry = {
                        "task_id": task_id,
                        "model": model_name,
                        "scenario": scenario,
                        "output": str(e),
                        "status": "error"
                    }
                    outputs = {}

                with open(results_file, 'a') as f:
                    f.write(json.dumps(result_entry) + "\n")

                if result_entry.get("status") == "success":
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
