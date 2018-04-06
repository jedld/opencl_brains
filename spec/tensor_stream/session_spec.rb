require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe TensorStream::Session do
  context "#run" do
    it "can execute operations" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream.constant(4.0) # also tf.float32 implicitly
      c = TensorStream.constant(5.0)
      total = a + b
      product = a * c
      sess = TensorStream.Session
      expect(sess.run(total).ruby_eval).to eq(7.0)
      expect(sess.run(product).ruby_eval).to eq(15.0)
    end
  end
end