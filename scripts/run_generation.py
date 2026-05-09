from pathlib import Path

from dotenv import load_dotenv

from ipf_rag.experiments.generation import generate_answers


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    output_path = project_root / "outputs" / "ipf_rag_generation_results.json"
    generate_answers(project_root=project_root, output_path=output_path, top_k=1)
