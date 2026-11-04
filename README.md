# Algoritmul Cuantic Barbu (Modelul Sigma-PDE)

Implementare de referință a **Algoritmului Cuantic Barbu**, o arhitectură de rețea neuronală cuantică (QNN 3D) și simulator de rețea elastodinamică derivat din **Modelul Sigma-PDE (Canonul Unificat v3)**.

* **Autor:** Ilie Barbu (ORCID: [0009-0005-3302-4417](https://orcid.org/0009-0005-3302-4417))
* **Cadru teoretic & DOI Zenodo:** [10.5281/zenodo.23938325](https://doi.org/10.5281/zenodo.23938325)

---

## Caracteristici Arhitecturale
1. **Topologie Wigner-Seitz 3D:** Mapare celulară cu $F = 6$ qubiți/celulă ($D=3, E=12, \chi=2$), reflectând tensorul de forfecare Navier-Cauchy al vidului.
2. **Scalabilitate $O(1)$ a Adâncimii:** Partiționarea muchiilor interne în 4 straturi paralele ($M_1..M_4$) și independența spațială a cuplajelor $X, Y, Z$ mențin adâncimea pe doi qubiți fixă (11 straturi), indiferent de numărul de celule.
3. **Atenuarea Barren Plateaus:** Entanglementul limitat de legea de arie a celulei previne dispersia exponențială a gradientului.
4. **Validare Experimentală:** Testat pe procesoare cuantice IBM Heron r2 (`ibm_kingston`, `ibm_marrakesh`, 48 qubiți, 1024 shots).

---

## Structura Proiectului
* `barbu_qnn.py`: Motorul algebric complet Sigma-PDE, generatorul de circuite OpenQASM și stratul neuronal Barbu-QNN.
* `job-dbl10jl6p51s73cnid20-result.json`: Rezultate experimentale brute (1024 măsurători pe 48 qubiți) rulate pe `ibm_marrakesh`.

---

## Rulare Rapidă

Pentru a rula calculul analitic și demonstrația QNN:

```bash
python barbu_qnn.py
