# Learnings
1. Got to know that state dimensions are equal to the number of units passed in the LSTM or so.

2. I had a misconception that in Badhnau that the alpha will also be a vector but its not like that. I got to know while coding

3. The Trade-off: Exposure Bias
    If you use only a True/False flag (100% teacher forcing during training, 0% during inference), your model suffers from something called Exposure Bias.

    Because the model was heavily spoon-fed the perfect answers during training, the moment it makes a single mistake during inference, it panics and the rest of the sentence turns into gibberish. It never learned how to recover from its own bad predictions.

4. I got to know about the TensorDataset method which is almost the fastest and easiest method which I encountered till now.