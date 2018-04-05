require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe TensorStream::Session do
  describe "Tensors" do
    it "creates tensor operations" do
      a = TensorStream::Tensor.constant(3.0, dtype: :float32)
      b = TensorStream::Tensor.constant(4.0)
      total = a + b
      expect(a.to_s).to eq("Const:0")
      expect(b.to_s).to eq("Const_1:0")
      expect(total.to_s).to eq("add:0")
    end
  end
end