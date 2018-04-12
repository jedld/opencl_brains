require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe TensorStream::Tensor do

  before(:each) do
    described_class.reset_counters
    TensorStream::Operation.reset_counters
    TensorStream::Graph.create_default
  end

  describe "Tensors" do
    it "can define Rank 0 Tensor definitions" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream.constant(4.0)
      c = TensorStream.constant(4.0)
      d = TensorStream.Variable(451, TensorStream::Types.int16)
      e = TensorStream.Variable(451.12)
      total = a + b + c
      # expect(TensorStream::Graph.get_default_graph.nodes.keys).to eq([])
      expect(a.to_s).to eq("Const:0")
      expect(b.to_s).to eq("Const_1:0")
      expect(c.to_s).to eq("Const_2:0")
      expect(total.to_s).to eq("add_1:0")
      expect(d.to_s).to eq("Variable:0")
      expect(e.to_s).to eq("Variable_2:0")
      expect(a.shape.to_s).to eq("TensorShape([])")
    end

    it "can define Rank 1 Tensor definitions" do
      a = TensorStream.constant([3.0], dtype: TensorStream::Types.float32)
      b = TensorStream.constant([])

      expect(a.to_s).to eq("Const:1")
      expect(a.shape.to_s).to eq("TensorShape([Dimension(1)])")
      expect(b.shape.to_s).to eq("TensorShape([Dimension(0)])")
    end
  end

  describe "#rank" do
    it "correctly gives the rank" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream.constant([3.0], dtype: TensorStream::Types.float32)
      c = TensorStream.constant([[3.0],[1.0]])
      d = TensorStream.constant([[[3.0,2.0]],[[1.0]]], dtype: TensorStream::Types.float32)
      expect(a.rank).to eq(0)
      expect(b.rank).to eq(1)
      expect(c.rank).to eq(2)
      expect(c.shape.to_s).to eq("TensorShape([Dimension(2),Dimension(1)])")
      expect(d.shape.to_s).to eq("TensorShape([Dimension(2),Dimension(1),Dimension(2)])")
    end
  end

  describe "#[]" do
    it "access indexes" do
      b = TensorStream.constant([3.0], dtype: TensorStream::Types.float32)
      expect(b[0].to_s).to eq("slice:0")
    end
  end

  describe "#shape" do
    it "gives the shape of the tensor" do
      b = TensorStream.constant([3.0], dtype: TensorStream::Types.float32)
      expect(b.shape[0]).to eq(1)
    end
  end

  describe "#dtype" do
    it "returns the tensor's datatype" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.int16)
      b = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      expect(a.dtype).to eq(:int16)
      expect(b.dtype).to eq(:float32)
    end
  end

  describe "#eval" do
    it "evaluates a tensor" do
      constant = TensorStream.constant([1, 2, 3])
      tensor = constant * constant
      expect(tensor.eval()).to eq([1, 4, 9])
    end
  end

  describe "#ruby_eval" do
    it "evaluates tensor to its ruby equivalent value" do
      a = TensorStream.constant([3.0, 1.0], dtype: TensorStream::Types.float32)
      expect(a.ruby_eval).to eq([3.0, 1.0])
    end
  end

  describe "naming operation" do
    it "can specify a name" do
      c_0 = TensorStream.constant(0, name: "c")
      expect(c_0.name).to eq("c")

      c_1 = TensorStream.constant(2, name: "c")
      expect(c_1.name).to eq("c_1")
    end
  end
end