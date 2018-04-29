require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::NN do
  let(:tf) { TensorStream } # Tensorflow compatibility alias

  context ".softmax" do
    it "computes for the softmax of a group of values" do
      outputs = tf.constant([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      expect(tr(tf.nn.softmax(outputs).eval)).to eq([0.0236, 0.0643, 0.1747, 0.4748, 0.0236, 0.0643, 0.1747])
    end

    specify "gradients" do
      outputs = tf.constant([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.nn.softmax(outputs)
      expect(tf.gradients(f, [outputs]).eval).to eq([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    end
  end
end