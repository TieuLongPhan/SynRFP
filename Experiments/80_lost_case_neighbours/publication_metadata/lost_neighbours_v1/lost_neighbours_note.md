# Nearest training neighbours of the Case B losses

All **2,308 eligible losses** were evaluated on their Request 75 display seeds, with **2,308 distinct both-correct controls** sampled in matching endpoint and split counts. The five held-out votes, saved predictions, training-set identities and 72 published examples were replayed. The search used optimizer-training rows and excluded the internal validation rows. No models were fitted.

Nearest pooled neighbours explain a minority of these errors under this diagnostic. D+U's incorrect prediction agrees with the nearest-by-U label in **30.6%** of losses and with the nearest-by-D+U label in **28.9%**. By comparison, D+U agrees with the nearest-by-U label in **72.7%** of both-correct controls. In **74.6%** of losses the nearest U label differs from the actual class. Within this subset, **41.1%** match the wrong prediction. Allowing any exact top-score tie raises the loss-set U agreement to **30.9%**.

For losses the median U similarity to the nearest U neighbour was **0.397**, with median D similarity **0.255** to those same neighbours. The median D similarity to the nearest D neighbour was **0.486**, with median U similarity **0.289** to those neighbours.

Controls show the prevalence of correct neighbour agreement when both models succeed. The diagnostic uses joint two-side pooled context and exact Tanimoto within the archived sketches. It cannot localize a signal to substrates alone or establish that the neighbour caused the MLP prediction. The D model used 2,048 bits. Nearest-by-D uses the 1,024-bit D block of D+U, as specified. Equal scores choose the lowest source position, then the lowest panel position. Two all-zero sketches have similarity 1. Archived zero placeholders are retained in fit candidate sets and flagged.

The loss CSV includes neighbour identifiers, match indicators and similarities. The 72 previously shown cases also include the nearest U reaction and its class for drawing. Scores for the reference reactions are reported in aggregate.
