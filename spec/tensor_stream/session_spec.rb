require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe TensorStream::Session do
  context "#run" do
    it "can execute operations" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream.constant(4.0) # also tf.float32 implicitly
      total = a + b

      sess = TensorStream.Session
      expect(sess.run(total).ruby_eval).to eq(7.0)
    end
  end
end