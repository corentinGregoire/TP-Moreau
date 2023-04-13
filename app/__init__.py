from fastapi import Form
from pydantic import BaseModel


class MaladieData(BaseModel):
    age: int
    genre: bool  # homme => 1 ; femme => 0
    taille: int
    poids: float
    pression_systo: int
    pression_diasto: int
    cholesterol: str
    glycemie: int
    fumeur: bool
    conso_alco: bool
    activite_physique: bool

    @classmethod
    def as_form(
            cls,
            age: int = Form(...),
            genre: bool = Form(...),
            taille: int = Form(...),
            poids: float = Form(...),
            pression_systo: int = Form(...),
            pression_diasto: int = Form(...),
            cholesterol: str = Form(...),
            glycemie: int = Form(...),
            fumeur: bool = Form(...),
            conso_alco: bool = Form(...),
            activite_physique: bool = Form(...)
    ):
        return cls(
            age=age,
            genre=genre,
            taille=taille,
            poids=poids,
            pression_systo=pression_systo,
            pression_diasto=pression_diasto,
            cholesterol=cholesterol,
            glycemie=glycemie,
            fumeur=fumeur,
            conso_alco=conso_alco,
            activite_physique=activite_physique
        )
