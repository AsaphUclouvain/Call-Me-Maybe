import json
from pydantic import BaseModel, ValidationError
from pathlib import Path
from typing import Literal


class VarMetaData(BaseModel):
    type: Literal["number", "string"]


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, VarMetaData]
    returns: VarMetaData


def load_function_from_json(file_path: str) -> list[dict[str, str | dict]]:
    path = Path(file_path)
    try:
        text = path.read_text(encoding="utf-8")
        raw_data = json.loads(text)
        if not isinstance(raw_data, list):
            raise ValueError(f"le fichier {file_path} doit contenir une liste JSON.")
        return [FunctionDef(**f) for f in raw_data]
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as e:
        raise ValueError(
                f"Impossible de charger les donnees depuis {file_path}: {e}"
            ) from e

print(load_function_from_json("data/input/functions_definition.json"))

"""
function definition is inserted in the prompt as text

[Prompt Utilisateur + Définition Fonctions + format_response] 
                  │
                  ▼
          1. sdk.encode()
                  │
                  ▼
         [final_prompt_ids] ◄──────────────────────────────────────┐
                  │                                                │
                  ▼                                                │ (Boucle)
       2. sdk.get_logits_from_input_ids()                          │
                  │                                                │
                  ▼                                                │
           [vocab_logits]                                          │
                  │                                                │
                  ▼                                                │
       3. TON CODE : Application du Masque                         │
   (Force à -∞ les tokens JSON non valides)                        │
                  │                                                │
                  ▼                                                │
       4. TON CODE : Échantillonnage (Sampling)                    │
        (Sélectionne le meilleur token)                            │
                  │                                                │
                  ▼                                                │
           [prochain_token_id] ────────────────────────────────────┘
                  │
                  ▼ (Quand le JSON est fermé : "}")
           5. sdk.decode()
                  │
                  ▼
         [Fichier JSON Final]

À chaque tour de boucle, le prochain_token_id choisi à l'étape 4 est ajouté (append) à la liste final_prompt_ids.

FORMAT_RESPONSE = {"name": "...", "parameters": {...}}
Nous allons ensuite inserer le prompt nous meme

On va appliquer le masque differemment entre les etapes en fonction
d'ou on en est dans le schema. 

recherche des candidats pour EXPECT_FUNCTION_NAME:

candidats_restants = [tous les noms de functions_definition.json]
matched_so_far = ""

prendre le meilleurs token directement revient a s'en remettre au model ce qui est interdit

tant que len(candidats_restants) > 1 (ou état pas encore terminal):
    1. logits = sdk.get_logits_from_input_ids(current_ids)
    2. tokens_valides = { id : token_str
                           pour chaque id du vocab
                           si un candidat de candidats_restants
                           commence par (matched_so_far + token_str) }
    3. meilleur_id = argmax(logits[id] pour id dans tokens_valides)   ← "meilleur score" ICI, à CE tour
    4. matched_so_far += id_to_token[meilleur_id]
    5. current_ids.append(meilleur_id)
    6. candidats_restants = [c pour c dans candidats_restants si c.startswith(matched_so_far)]

"""