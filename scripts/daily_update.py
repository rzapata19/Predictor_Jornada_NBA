from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(script_name: str, label: str) -> None:
    script_path = SCRIPTS_DIR / script_name
    print(f"\n[{label}] Ejecutando {script_name}...")
    result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Fallo en {script_name} con codigo {result.returncode}.")


def main() -> None:
    print("Actualizacion diaria del proyecto NBA")
    print("1. Regenerar dataset")
    print("2. Actualizar lesiones")
    print("3. Reentrenar modelo")
    print("4. Generar predicciones de las proximas 24 horas")

    run_step("build_dataset.py", "Paso 1")
    run_step("update_injuries.py", "Paso 2")
    run_step("train_model.py", "Paso 3")
    run_step("predict_next_24h.py", "Paso 4")

    print("\nProceso completado.")


if __name__ == "__main__":
    main()
