import os
import re
import shutil
from collections import defaultdict


def comparar_modelos_por_id_com_consistencia(
    exp1_base,
    exp1_modelo,
    exp2_base,
    exp2_modelo,
    mensagem="mensagem_comparacao:",
    output_dir="Comparacao",
    salvar_mapas=False,
    dir_mapas=None,
):
    """
    Compara modelos por ID considerando consistencia entre mapas.
    """

    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    def only_id(name: str) -> str:
        match = re.search(r"(id_\d+)", name, flags=re.IGNORECASE)
        return match.group(1) if match else name

    def listar_pngs(path):
        return [f for f in os.listdir(path) if f.lower().endswith(".png")]

    def coletar_por_experimento(exp_base: str, modelo: str):
        por_id = {
            "Health": defaultdict(lambda: {"Acertos": set(), "Erros": set()}),
            "Sick": defaultdict(lambda: {"Acertos": set(), "Erros": set()}),
        }
        totais_arquivos = {
            "Health": {"Acertos": 0, "Erros": 0},
            "Sick": {"Acertos": 0, "Erros": 0},
        }

        for classe in ["Health", "Sick"]:
            for grupo in ["Acertos", "Erros"]:
                class_path = os.path.join(exp_base, grupo, modelo, classe)
                if not os.path.exists(class_path):
                    continue

                arquivos = listar_pngs(class_path)
                totais_arquivos[classe][grupo] += len(arquivos)

                for fname in arquivos:
                    _id = only_id(fname)
                    por_id[classe][_id][grupo].add(fname)

        return por_id, totais_arquivos

    def rotular_consistencia(entry: dict):
        acertos = len(entry["Acertos"])
        erros = len(entry["Erros"])
        total = acertos + erros

        if total == 0:
            return "AUSENTE", None, {"Acertos": acertos, "Erros": erros}, total
        if acertos > 0 and erros > 0:
            return "MISTO", None, {"Acertos": acertos, "Erros": erros}, total
        if acertos > 0:
            return "CONSISTENTE", "Acertos", {"Acertos": acertos, "Erros": erros}, total
        return "CONSISTENTE", "Erros", {"Acertos": acertos, "Erros": erros}, total

    if dir_mapas is None:
        dir_mapas = os.path.join(output_dir, "mapas_calor")

    def copiar_mapas_id(classe: str, categoria: str, e1_entry: dict, e2_entry: dict, g1: str, g2: str):
        dest_exp1 = os.path.join(dir_mapas, classe, categoria, "exp1")
        dest_exp2 = os.path.join(dir_mapas, classe, categoria, "exp2")
        ensure_dir(dest_exp1)
        ensure_dir(dest_exp2)

        for fname in e1_entry[g1]:
            src = os.path.join(exp1_base, g1, exp1_modelo, classe, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_exp1, fname))

        for fname in e2_entry[g2]:
            src = os.path.join(exp2_base, g2, exp2_modelo, classe, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_exp2, fname))

    exp1_por_id, exp1_totais = coletar_por_experimento(exp1_base, exp1_modelo)
    exp2_por_id, exp2_totais = coletar_por_experimento(exp2_base, exp2_modelo)

    ensure_dir(output_dir)
    relatorio_path = os.path.join(output_dir, "relatorio_comparacao.txt")
    with open(relatorio_path, "w", encoding="utf-8") as f:
        if mensagem == "mensagem_comparacao:":
            f.write(f"Comparando o modelo {exp1_modelo} com {exp2_modelo}\n")
        else:
            f.write(mensagem.strip() + "\n")

    with open(relatorio_path, "a", encoding="utf-8") as report:
        for classe in ["Health", "Sick"]:
            report.write(
                f"\n============================================= {classe} =============================================\n"
            )
            report.write(
                "\n---------------------------- Quantitativo total (por ARQUIVO) ----------------------------\n"
            )
            report.write(
                f"Modelo 1 - Acertos: {exp1_totais[classe]['Acertos']} | Erros: {exp1_totais[classe]['Erros']}\n"
            )
            report.write(
                f"Modelo 2 - Acertos: {exp2_totais[classe]['Acertos']} | Erros: {exp2_totais[classe]['Erros']}\n"
            )

            ids_classe = set(exp1_por_id[classe].keys()) | set(exp2_por_id[classe].keys())
            melhorou = []
            piorou = []
            manteve_ac = []
            manteve_er = []
            revisao_manual = []
            ausentes = []

            for _id in sorted(
                ids_classe,
                key=lambda x: (int(re.search(r"\d+", x).group()), x) if re.search(r"\d+", x) else (10**9, x),
            ):
                e1_entry = exp1_por_id[classe].get(_id, {"Acertos": set(), "Erros": set()})
                e2_entry = exp2_por_id[classe].get(_id, {"Acertos": set(), "Erros": set()})

                s1, g1, c1, _t1 = rotular_consistencia(e1_entry)
                s2, g2, c2, _t2 = rotular_consistencia(e2_entry)

                if s1 == "AUSENTE" or s2 == "AUSENTE":
                    ausentes.append((_id, s1, s2, c1, c2))
                    continue

                if s1 == "MISTO" or s2 == "MISTO":
                    detalhe_exp1 = f"exp1: Acertos={c1['Acertos']}, Erros={c1['Erros']}"
                    detalhe_exp2 = f"exp2: Acertos={c2['Acertos']}, Erros={c2['Erros']}"
                    revisao_manual.append((_id, detalhe_exp1, detalhe_exp2))
                    continue

                categoria = None
                if g1 == "Erros" and g2 == "Acertos":
                    melhorou.append(_id)
                    categoria = "Erro_Acerto"
                elif g1 == "Acertos" and g2 == "Erros":
                    piorou.append(_id)
                    categoria = "Acerto_Erro"
                elif g1 == "Acertos" and g2 == "Acertos":
                    manteve_ac.append(_id)
                    categoria = "Manteve_Acerto"
                elif g1 == "Erros" and g2 == "Erros":
                    manteve_er.append(_id)
                    categoria = "Manteve_Erro"

                if salvar_mapas and categoria is not None:
                    copiar_mapas_id(
                        classe=classe,
                        categoria=categoria,
                        e1_entry=e1_entry,
                        e2_entry=e2_entry,
                        g1=g1,
                        g2=g2,
                    )

            report.write(
                "\n---------------------------- Comparação (apenas IDs CONSISTENTES em ambos) ----------------------------\n"
            )
            report.write(f"Erro -> Acerto (melhorou): {len(melhorou)} IDs\n")
            report.write(f"Acerto -> Erro (piorou):   {len(piorou)} IDs\n")
            report.write(f"Manteve_Acerto:            {len(manteve_ac)} IDs\n")
            report.write(f"Manteve_Erro:              {len(manteve_er)} IDs\n\n")

            def listar(titulo, items):
                report.write(f"--- {titulo} ---\n")
                for item in items:
                    report.write(f"{item}\n")
                report.write("\n")

            listar("Erro -> Acerto (melhorou)", melhorou)
            listar("Acerto -> Erro (piorou)", piorou)
            listar("Manteve_Acerto", manteve_ac)
            listar("Manteve_Erro", manteve_er)

            report.write(
                "---------------------------- IDs para REVISÃO MANUAL (mistos em algum experimento) ----------------------------\n"
            )
            report.write(f"Total: {len(revisao_manual)} IDs\n")
            for _id, d1, d2 in revisao_manual:
                report.write(f"{_id} | {d1} | {d2}\n")
            report.write("\n")

            report.write(
                "---------------------------- IDs AUSENTES (presentes em apenas 1 experimento) ----------------------------\n"
            )
            report.write(f"Total: {len(ausentes)} IDs\n")
            for _id, s1, s2, c1, c2 in ausentes:
                report.write(
                    f"{_id} | exp1={s1} (A={c1['Acertos']},E={c1['Erros']}) | exp2={s2} (A={c2['Acertos']},E={c2['Erros']})\n"
                )
            report.write("\n")

    print("Comparação concluída!")
    print(f"Relatório: {relatorio_path}")
    print(f"Pasta de saída: {output_dir}")
