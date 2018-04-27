module TensorStream
  class NN
    def self.softmax(logits, options = {})
      TensorStream.exp(logits) / TensorStream.reduce_sum(TensorStream.exp(logits))
    end
  end

  # tensorflow compatibility
  def self.nn
    TensorStream::NN
  end
end