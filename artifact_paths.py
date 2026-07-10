from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIRNAME = "resultados"
JSON_DIRNAME = "json"
MATRICES_DIRNAME = "matrizes"


def _resolve_base_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is None:
        return ROOT_DIR
    path = Path(base_dir)
    if path.is_absolute():
        return path.resolve()
    return (ROOT_DIR / path).resolve()


def json_dir(base_dir: str | Path | None = None) -> Path:
    return _resolve_base_dir(base_dir) / RESULTS_DIRNAME / JSON_DIRNAME


def matrices_dir(base_dir: str | Path | None = None) -> Path:
    return _resolve_base_dir(base_dir) / RESULTS_DIRNAME / MATRICES_DIRNAME


def ensure_output_dirs(base_dir: str | Path | None = None) -> tuple[Path, Path]:
    json_output_dir = json_dir(base_dir)
    matrices_output_dir = matrices_dir(base_dir)
    json_output_dir.mkdir(parents=True, exist_ok=True)
    matrices_output_dir.mkdir(parents=True, exist_ok=True)
    return json_output_dir, matrices_output_dir


def json_output_path(filename: str, base_dir: str | Path | None = None) -> Path:
    json_output_dir, _ = ensure_output_dirs(base_dir)
    return json_output_dir / Path(filename).name


def matrix_output_path(filename: str, base_dir: str | Path | None = None) -> Path:
    _, matrices_output_dir = ensure_output_dirs(base_dir)
    return matrices_output_dir / Path(filename).name


def _artifact_candidates(
    filenames: tuple[str, ...],
    output_dir_resolver,
) -> list[Path]:
    output_dir = output_dir_resolver()
    candidates: list[Path] = []
    seen: set[str] = set()

    for filename in filenames:
        legacy_path = ROOT_DIR / filename
        current_path = output_dir / Path(filename).name
        for path in (current_path, legacy_path):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)
    return candidates


def json_candidates(*filenames: str) -> list[Path]:
    return _artifact_candidates(filenames, json_dir)


def matrix_candidates(*filenames: str) -> list[Path]:
    return _artifact_candidates(filenames, matrices_dir)


def display_path(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path_obj)
