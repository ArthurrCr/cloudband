# Limitações

O que fica em aberto nos resultados e precisa ser declarado ao reportá-los. A origem de cada item está em
`AUDIT.md` ou em `protocols/DIVERGENCES.md`.

---

## L1. Treino e validação com o recorte antigo do patch de 509 pixels

**Origem:** AUDIT A1, DIVERGENCES D17. **Decisão de 15/11/2026:** os modelos não são retreinados.

As 15 execuções de treino da fase 2 (5 seeds de regnety, convnext e Swin-T + UPerNet), a busca de taxa de
aprendizado e a seleção do melhor checkpoint por entropia cruzada de validação usaram patches recortados com o
preenchimento do lado errado: cada patch mantinha 3 colunas de preenchimento (imagem zerada, rótulo céu limpo;
1.527 pixels, 0,59% do patch) e perdia as suas 3 primeiras colunas reais.

**Por que os resultados continuam válidos para a comparação:**

- As duas arquiteturas treinaram e foram selecionadas exatamente sobre os mesmos patches. O defeito é uma
  característica comum dos dados de treino, não uma diferença entre elas.
- O teste é lido com o recorte corrigido: só pixels reais e anotados entram na conta, e a reação de um modelo a uma
  entrada zerada não pesa em nenhum resultado.
- As versões publicadas do OmniCloudMask são avaliadas nos mesmos patches corrigidos, e não ganham mais com a faixa.

**O que não se pode afirmar:** que os modelos treinados teriam o mesmo desempenho se treinados com o recorte
correto. O efeito esperado é pequeno (0,59% dos pixels), mas não foi medido.

**Texto sugerido para a metodologia:** "Os modelos foram treinados com patches de 509 × 509 pixels recortados do
canvas de 512 com o preenchimento do lado errado em um eixo, o que manteve 3 colunas de preenchimento rotuladas
como céu limpo e removeu 3 colunas reais (0,59% de cada patch), igualmente para as duas arquiteturas. A avaliação
no conjunto de teste usou o recorte correto."

**Resultados que não devem ser usados:** tudo o que foi calculado antes da correção, ou seja, os arquivos de
contagem sem `crop2` no nome e as tabelas e figuras feitas a partir deles.
