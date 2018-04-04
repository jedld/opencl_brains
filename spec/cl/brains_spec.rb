require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe Cl::Brains do
  it "has a version number" do
    expect(Cl::Brains::VERSION).not_to be nil
  end

  def generate_matrix(size)
    arr = []
    size.times do |row|
      row_arr = []
      size.times do |col|
        row_arr << rand(100)
      end
      arr << row_arr
    end
    Cl::Brains::Tensor.matrix(arr)
  end
  
  context "sigmoid operations" do
    let(:matrix_a) {
      Cl::Brains::Tensor.matrix([
       [ -3.0,	4.0,	6.0],
       [ -2.0,	-1.0, -4.0],
       [  3.0,	0.0,  5.0]
      ])
    }

    it "4x4 Sigmoid computation" do
      compute = Cl::Brains::CLMatrixMath.new(3)
      puts Benchmark.measure {
        expect(compute.sigmoid([[-1, 2],[1, -2]]).execute.rubynize).to eq([[0.2689414322376251, 0.8807970285415649], [0.7310585975646973, 0.11920291930437088]])
      }
    end
  end

  context "Matrix operations" do
    let(:matrix_a) {
      Cl::Brains::Tensor.matrix([
       [ -3.0,	4.0,	6.0],
       [ -2.0,	-1.0, -4.0],
       [  3.0,	0.0,  5.0]
      ])
    }

    let(:matrix_b) {
      Cl::Brains::Tensor.matrix([
       [1.0,	2.0,	-3.0],
       [4.0,	6.0,	-4.0],
       [-2.0,	0.0,   3.0]
      ])
    }

    let(:matrix_x) {
      Cl::Brains::Tensor.matrix([
        [1.0],
        [2.0],
        [-3.0]
      ])
    }

    it "4x4 Matrix multiplication" do
      compute = Cl::Brains::CLMatrixMath.new(3)
      puts Benchmark.measure {
        expect(compute.mul(Cl::Brains::Tensor.matrix([[-1, 2],[1, -2]]), Cl::Brains::Tensor.matrix([[2, -2],[-1, 2]])).execute.rubynize).to eq([[-4, 6], [4, -6]])
      }
      puts Benchmark.measure {
        expect((Matrix[*[[-1, 2],[1, -2]]] *  Matrix[*[[2, -2],[-1, 2]]]).to_a).to eq([[-4, 6], [4, -6]])
      }
    end

    it "Matrix multiplication" do
      compute = Cl::Brains::CLMatrixMath.new(3)
      puts Benchmark.measure {
        expect(compute.mul(matrix_a, matrix_b).execute.rubynize).to eq([
          [1.0,	18.0,	11.0],
          [2.0,	-10.0, -2.0],
          [-7.0,	6.0,	6.0]
        ])
      }
      puts Benchmark.measure {
      expect((Matrix[*matrix_a] *  Matrix[*matrix_b]).to_a).to eq([
        [1.0,	18.0,	11.0],
        [2.0,	-10.0, -2.0],
        [-7.0,	6.0,	6.0]
      ])
      }
    end

    it "Matrix multiplication 2" do
      compute = Cl::Brains::CLMatrixMath.new(3)
      expect((Matrix[*matrix_a] *  Matrix[*matrix_x]).to_a).to eq( [[-13.0], [8.0], [-12.0]])
      expect(compute.mul(matrix_a, matrix_x).execute.rubynize).to eq(
        [
          [-13.0],
          [8.0],
          [-12.0]
        ])
    end

    it "large matrices" do
      ma = generate_matrix(512)
      mb = generate_matrix(512)
      expected = nil
      ap "Using CPU"
      puts Benchmark.measure {
        expected = (Matrix[*ma] *  Matrix[*mb]).to_a
      }
      ap  "Using GPU"
      compute = Cl::Brains::CLMatrixMath.new(512)
      puts Benchmark.measure {
        expect(compute.mul(ma, mb).execute.rubynize).to eq(expected)
      }
    end
  end
end
