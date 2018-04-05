require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe TensorStream::Tensor do

  before(:each) do
    described_class.reset_counters
    TensorStream::Operation.reset_counters
  end

  describe "Tensors" do
    it "can define Rank 0 Tensor definitions" do
      a = TensorStream::Tensor.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream::Tensor.constant(4.0)
      c = TensorStream::Tensor.constant(4.0)
      d = TensorStream::Tensor.Variable(451, TensorStream::Types.int16)
      e = TensorStream::Tensor.Variable(451.12)
      total = a + b + c
      expect(a.to_s).to eq("Const:0")
      expect(b.to_s).to eq("Const_1:0")
      expect(c.to_s).to eq("Const_2:0")
      expect(total.to_s).to eq("add_1:0")
      expect(d.to_s).to eq("Variable:0")
      expect(e.to_s).to eq("Variable_2:0")
      expect(a.shape.to_s).to eq("TensorShape([])")
    end

    it "can define Rank 1 Tensor definitions" do
      a = TensorStream::Tensor.constant([3.0], dtype: TensorStream::Types.float32)
      b = TensorStream::Tensor.constant([])

      expect(a.to_s).to eq("Const:1")
      expect(a.shape.to_s).to eq("TensorShape([Dimension(1)])")
      expect(b.shape.to_s).to eq("TensorShape([Dimension(0)])")
    end
  end

  describe "#rank" do
    it "correctly gives the rank" do
      a = TensorStream::Tensor.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream::Tensor.constant([3.0], dtype: TensorStream::Types.float32)
      c = TensorStream::Tensor.constant([[3.0],[1.0]])
      d = TensorStream::Tensor.constant([[[3.0,2.0]],[[1.0]]], dtype: TensorStream::Types.float32)
      expect(a.rank).to eq(0)
      expect(b.rank).to eq(1)
      expect(c.rank).to eq(2)
      expect(c.shape.to_s).to eq("TensorShape([Dimension(2),Dimension(1)])")
      expect(d.shape.to_s).to eq("TensorShape([Dimension(2),Dimension(1),Dimension(2)])")
    end
  end

  describe "#[]" do
    it "access indexes" do
      b = TensorStream::Tensor.constant([3.0], dtype: TensorStream::Types.float32)
      expect(b[0].to_s).to eq("slice:0")
    end
  end

  describe "#shape" do
    it "gives the shape of the tensor" do
      b = TensorStream::Tensor.constant([3.0], dtype: TensorStream::Types.float32)
      expect(b.shape[0]).to eq(1)
    end
  end
end