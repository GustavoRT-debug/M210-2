# 📘 **Simplex com Análise de Sensibilidade – Streamlit App**

Este projeto implementa o **Método Simplex** para Programação Linear, incluindo **Análise de Sensibilidade completa**, tudo em uma interface web interativa desenvolvida com **Streamlit**.

---

## 🚀 **Funcionalidades**

### ✅ **1. Resolução de Problemas de Programação Linear**

* Maximização da função objetivo
* Restrições do tipo **≤** ou **≥**
* Conversão automática para forma padrão

### ✅ **2. Método Simplex Completo**

* Tableau do Simplex montado automaticamente
* Pivotamento linha a linha
* Verificação de:

  * Solução ótima
  * Problema ilimitado
  * Solução inviável

### ✅ **3. Análise de Sensibilidade**

Para cada restrição, o sistema calcula:

* **Preço-sombra (dual)**
* **Variação permitida de bᵢ** → *(delta_min, delta_max)*
* **Intervalo completo permitido para bᵢ**
* Conversão correta para o modelo original mesmo com desigualdades ≥

### ✅ **4. Variação Simultânea dos Recursos**

* Permite testar um **vetor Δ simultâneo**
* Verifica:

  * Se a mesma base permanece viável
  * Novo lucro estimado: `z' = z + π · Δ`

### ✅ **5. Interface Gráfica via Streamlit**

* Campos dinâmicos para número de variáveis e restrições
* Tabelas formatadas com pandas
* Exibição clara dos resultados

---

## 🛠 **Tecnologias Utilizadas**

| Tecnologia                   | Uso                                   |
| ---------------------------- | ------------------------------------- |
| **Python**                   | Lógica matemática e processamento     |
| **NumPy**                    | Manipulação matricial                 |
| **Pandas**                   | Tabelas de sensibilidade              |
| **Streamlit**                | Interface web                         |
| **Simplex**                  | Algoritmo de otimização               |
| **Análise de Sensibilidade** | Cálculo de preços-sombra e intervalos |

---

## 📦 **Instalação**

1. Clone este repositório:

```bash
git clone https://github.com/seu_usuario/seu_repositorio.git
cd seu_repositorio
```

2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

> Caso você não tenha o arquivo `requirements.txt`, ele deve conter:

```txt
numpy
pandas
streamlit
```

---

## ▶️ **Como executar o projeto**

Execute este comando na pasta do projeto:

```bash
streamlit run app.py
```

Ou substitua `app.py` pelo nome do seu arquivo principal.

---

## 📊 **Como Usar a Aplicação**

### 1. Informe:

* Número de variáveis
* Número de restrições

### 2. Digite:

* Coeficientes da função objetivo
* Cada restrição (A, sinal ≤/≥ e b)

### 3. Clique em **Resolver**

O sistema exibirá:

* Solução ótima `x*`
* Valor ótimo `z*`
* Tabela de:

  * Preços-sombra
  * Intervalos de sensibilidade de cada bᵢ

### 4. Para testar **variação simultânea Δ**:

* Preencha Δ₁, Δ₂, …
* Clique em **Testar variação**

---

## 🧠 **Funcionamento Interno (Resumo)**

* O sistema normaliza restrições (≥ → multiplica por -1).
* Constrói o tableau completo.
* Executa pivotamento com regra do mínimo razão.
* Calcula:

  * Solução básica
  * Matriz da base B
  * Inversa B⁻¹
  * Vetor de preços-sombra π = cᵦᵀB⁻¹
* Determina intervalos de sensibilidade por análises em cada coluna de B⁻¹.
* Permite testar Δ simultâneo aplicando:
  [
  x_B' = B^{-1} (b + \Delta)
  ]
  Se todos x₍ᵦ₎ ≥ 0 → base viável.
---

## 📄 **Licença**

Este projeto está sob a licença MIT.
Você pode usá-lo livremente, inclusive para fins acadêmicos.

---

## 🙋 **Autores**
Matheus José Almeida Finamor, 
Luiz Henrique Vilas Boas da Silva, 
Gustavo Ramos L. Torres



