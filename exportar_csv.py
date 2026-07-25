"""
Gera os CSVs da base PEDE a partir das planilhas .xlsx publicadas no GitHub.
"""

from __future__ import annotations

import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parents[0]
PASTA_DESTINO = RAIZ / "Arquivos_Base"


EXPORTACOES: list[tuple[str, str, str]] = [
    ("BASE DE DADOS PEDE 2024", "PEDE2022", "BASE DE DADOS PEDE 2022"),
    ("BASE DE DADOS PEDE 2024", "PEDE2023", "BASE DE DADOS PEDE 2023"),
    ("BASE DE DADOS PEDE 2024", "PEDE2024", "BASE DE DADOS PEDE 2024"),
    ("BASE DE DADOS PEDE 2020_2022", "PEDE_PASSOS_DATASET_FIAP", "BASE DE DADOS PEDE 2020_2022")
]

def baixar(nome_xlsx: str) -> bytes:
    caminho = urllib.parse.quote(f"Arquivos_Base/{nome_xlsx}.xlsx")
    url = f"https://raw.githubusercontent.com/jnssena/datathon_2026/main/{caminho}"
    with urllib.request.urlopen(url, timeout=60) as resposta:
        return resposta.read()


# Normalização do formato das datas, removendo data e hora
def normalizar_datas(df: pd.DataFrame) -> pd.DataFrame:
    saida = df.copy()
    for coluna in saida.columns:
        if pd.api.types.is_datetime64_any_dtype(saida[coluna]):
            saida[coluna] = saida[coluna].dt.strftime("%Y-%m-%d")
    return saida


def converter() -> None:
    PASTA_DESTINO.mkdir(parents=True, exist_ok=True)

    for nome_xlsx, aba, nome_csv in EXPORTACOES:
        df = pd.read_excel(io.BytesIO(baixar(nome_xlsx)), sheet_name=aba)
        df = df.dropna(how="all")  # o Excel deixa uma linha vazia no fim
        df = normalizar_datas(df)

        destino = PASTA_DESTINO / f"{nome_csv}.csv"
        df.to_csv(destino, sep=";", decimal=",", encoding="utf-8-sig", index=False)
        print(destino.name)


def main() -> int:
    converter()
    return 0

if __name__ == "__main__":
    sys.exit(main())