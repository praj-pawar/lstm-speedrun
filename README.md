# readme: next word prediction speedrun 🚀

this jupyter notebook (`lstm_speedrun.ipynb`) serves as a quick, end-to-end example for learning **dataset preparation** and **pytorch implementation** for a basic natural language processing (nlp) task: **next word prediction**.

the model is trained on a small, provided text corpus (the "talktube privacy policy") to predict the next word in a sequence.

***

## 🎯 project goal (learn pytorch & data prep)

the primary purpose of this notebook is to provide a working template to understand the foundational steps required for pytorch sequence modeling:

1.  **text preprocessing:** tokenization and vocabulary building.
2.  **sequence generation:** creating input/target pairs (e.g., $w_1, \dots, w_{n-1} \to w_n$).
3.  **padding:** handling variable sequence lengths with `torch.nn.utils.rnn.pad_sequence`.
4.  **pytorch model construction:** defining an $\text{nn.module}$ with `nn.embedding` and $\text{nn.lstm}$.
5.  **training loop:** setting up the basic pytorch training cycle with $\text{adam}$ and $\text{crossentropyloss}$.

***

## 🧱 model architecture

the model uses a simple stacked long short-term memory (lstm) network:

| layer | function | key shape |
| :--- | :--- | :--- |
| **embedding** | maps words (indices) to vectors. | `(batch size, sequence length, 100)` |
| **lstm (x2)** | captures sequential dependencies. | intermediate sequence output. |
| **linear (dense)** | maps the final hidden state to a probability distribution. | `(batch size, 501)` |

***

## ⚠️ note on overfitting

due to the very small, closed corpus (the single privacy policy document), the model achieves $\mathbf{100\%}$ training accuracy quickly. this demonstrates **severe overfitting**.

* the **loss continues to drop** even at $100\%$ accuracy because the model is learning to be **more certain** about its already correct answers.
* in a real-world scenario, you would use a large dataset and techniques like **dropout** and a **validation set** to prevent this memorization.
