import argparse

from ragprobe.services.rag_server import create_app
from ragprobe.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Serve the RAG probe scorer (features -> trained fusion probe -> P(correct)).")
    parser.add_argument("--config", required=True)
    # migration overrides: swap probes / model / GPU / port without touching the config file
    parser.add_argument("--model", help="override model_name (HF id or local path)")
    parser.add_argument("--context-probe", help="override context_fusion_checkpoint path")
    parser.add_argument("--question-probe", help="override question_fusion_checkpoint path")
    parser.add_argument("--dual-probe", help="override dual_fusion_checkpoint path (q+ctx fusion head)")
    parser.add_argument("--gpu", help="override gpu_id (CUDA device index)")
    parser.add_argument("--port", type=int, help="override port")
    parser.add_argument("--host", help="override host")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.model:
        config["model_name"] = args.model
    if args.context_probe:
        config["context_fusion_checkpoint"] = args.context_probe
    if args.question_probe:
        config["question_fusion_checkpoint"] = args.question_probe
    if args.dual_probe:
        config["dual_fusion_checkpoint"] = args.dual_probe
    if args.gpu is not None:
        config["gpu_id"] = str(args.gpu)
    if args.port:
        config["port"] = args.port
    if args.host:
        config["host"] = args.host

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Please install fastapi and uvicorn to run the probe scorer service") from exc

    app = create_app(config)
    uvicorn.run(app, host=config.get("host", "0.0.0.0"), port=int(config.get("port", 8008)))


if __name__ == "__main__":
    main()
