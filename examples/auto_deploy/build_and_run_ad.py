"""Main entrypoint to build, test, and prompt AutoDeploy inference models."""

import argparse
import json
from typing import List, Optional, Union

import torch
from simple_config import SimpleConfig

from tensorrt_llm._torch.auto_deploy.models import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.shim import AutoDeployConfig, DemoLLM
from tensorrt_llm._torch.auto_deploy.utils.benchmark import benchmark
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi.llm import LLM, RequestOutput
from tensorrt_llm.sampling_params import SamplingParams

# Global torch config, set the torch compile cache to fix up to llama 405B
torch._dynamo.config.cache_size_limit = 20


def get_config_and_check_args() -> SimpleConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=json.loads)
    parser.add_argument("-m", "--model-kwargs", type=json.loads)
    args = parser.parse_args()
    configs_from_args = args.config or {}
    configs_from_args["model_kwargs"] = getattr(args, "model_kwargs") or {}
    config = SimpleConfig(**configs_from_args)
    ad_logger.info(f"Simple Config: {config}")
    return config


def build_llm_from_config(config: SimpleConfig) -> LLM:
    """Builds a LLM object from our config."""
    # set up builder config
    build_config = BuildConfig(max_seq_len=config.max_seq_len, max_batch_size=config.max_batch_size)
    build_config.plugin_config.tokens_per_block = config.page_size

    # setup AD config
    ad_config = AutoDeployConfig(
        # Both torch-opt and torch-cudagraph invoke cudagraphs
        use_cuda_graph=config.compile_backend in ["torch-opt", "torch-cudagraph"],
        # Both torch-opt and torch-compile invoke torch.compile
        torch_compile_enabled=config.compile_backend in ["torch-opt", "torch-compile"],
        model_factory=config.model_factory,
        model_kwargs=config.model_kwargs,
        attn_backend=config.attn_backend,
        mla_backend=config.mla_backend,
        skip_loading_weights=config.skip_loading_weights,
        cuda_graph_max_batch_size=config.max_batch_size,
    )
    ad_logger.info(f"AutoDeploy Config: {ad_config}")

    # TODO: let's see if prefetching can't be done through the LLM api?
    # I believe the "classic workflow" invoked via the LLM api can do that.
    # put everything into the HF model Factory and try pre-fetching the checkpoint
    factory = ModelFactoryRegistry.get(config.model_factory)(
        model=config.model,
        model_kwargs=config.model_kwargs,
        tokenizer_kwargs=config.tokenizer_kwargs,
        skip_loading_weights=config.skip_loading_weights,
    )
    ad_logger.info(f"Prefetched model : {factory.model}")

    # construct llm high-level interface object
    llm_lookup = {
        "demollm": DemoLLM,
        "trtllm": LLM,
    }
    llm = llm_lookup[config.runtime](
        model=factory.model,
        backend="autodeploy",
        build_config=build_config,
        pytorch_backend_config=ad_config,
        tensor_parallel_size=config.world_size,
        tokenizer=factory.init_tokenizer() if config.customize_tokenizer else None,
    )

    return llm


def print_outputs(outs: Union[RequestOutput, List[RequestOutput]]):
    if isinstance(outs, RequestOutput):
        outs = [outs]
    for i, out in enumerate(outs):
        ad_logger.info(f"[PROMPT {i}] {out.prompt}: {out.outputs[0].text}")


@torch.inference_mode()
def main(config: Optional[SimpleConfig] = None):
    if config is None:
        config = get_config_and_check_args()

    llm = build_llm_from_config(config)

    # prompt the model and print its output
    outs = llm.generate(
        config.prompt,
        sampling_params=SamplingParams(
            max_tokens=config.max_tokens,
            top_k=config.top_k,
            temperature=config.temperature,
        ),
    )
    print_outputs(outs)

    # run a benchmark for the model with batch_size == config.benchmark_bs
    if config.benchmark:
        keys = [
            "compile_backend",
            "attn_backend",
            "mla_backend",
            "benchmark_bs",
            "benchmark_isl",
            "benchmark_osl",
            "benchmark_num",
        ]
        benchmark(
            func=lambda: llm.generate(
                torch.randint(0, 100, (config.benchmark_bs, config.benchmark_isl)).tolist(),
                sampling_params=SamplingParams(max_tokens=config.benchmark_osl, top_k=None),
                use_tqdm=False,
            ),
            num_runs=config.benchmark_num,
            log_prefix="Benchmark with " + ", ".join(f"{k}={getattr(config, k)}" for k in keys),
            results_path=config.benchmark_results_path,
        )
    llm.shutdown()


if __name__ == "__main__":
    main()
