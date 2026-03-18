from __future__ import annotations

from pathlib import Path


REQUIRED_FILES = [
	"news.tsv",
	"behaviors.tsv",
	"entity_embedding.vec",
	"relation_embedding.vec",
]


def _line_count(path: Path) -> int:
	with path.open("r", encoding="utf-8") as handle:
		return sum(1 for _ in handle)


def main() -> None:
	data_dir = Path(__file__).resolve().parent
	print(f"Data directory: {data_dir}")

	missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
	if missing:
		print("Missing files:")
		for file_name in missing:
			print(f"- {file_name}")
		return

	print("All required files are present.")
	for file_name in REQUIRED_FILES:
		file_path = data_dir / file_name
		size_mb = file_path.stat().st_size / (1024 * 1024)
		if file_name.endswith(".tsv"):
			lines = _line_count(file_path)
			print(f"{file_name}: {lines} rows, {size_mb:.2f} MB")
		else:
			print(f"{file_name}: {size_mb:.2f} MB")


if __name__ == "__main__":
	main()
