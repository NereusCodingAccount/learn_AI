# Gated Recurrent Unit/GRU 閘門神經網路(自取)

### In artificial neural networks, the gated recurrent unit (GRU) is a gating mechanism used in recurrent neural networks, introduced in 2014 by Kyunghyun Cho et al. The GRU is like a long short-term memory (LSTM) with a gating mechanism to input or forget certain features, but lacks a context vector or output gate, resulting in fewer parameters than LSTM. GRU's performance on certain tasks of polyphonic music modeling, speech signal modeling and natural language processing was found to be similar to that of LSTM. GRUs showed that gating is indeed helpful in general, and Bengio's team came to no concrete conclusion on which of the two gating units was better.

## Architecture (建築學?)

### Fully gated unit
![alt text](../Pictures/GRU_1.png)

![alt text](../Pictures/GRU_2.png)
### Minimal gated unit

![alt text](../Pictures/GRU_3.png)

### Light gated recurrent unit

![alt text](../Pictures/GRU_4.png)

---

## References

1. https://en.wikipedia.org/wiki/Gated_recurrent_unit
2. https://web.archive.org/web/20211110112626/http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/


# [返回](../../ANN.md)