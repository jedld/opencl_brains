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
        row_arr << rand(1000)
      end
      arr << row_arr
    end
    arr
  end

  context "Matrix operations" do
    let(:matrix_a) {
      [
       [ -3,	4,	6],
       [ -2,	-1, -4],
       [  3,	0,  5]
      ]
    }

    let(:matrix_b) {
      [
       [1,	2,	-3],
       [4,	6,	-4],
       [-2,	0,   3]
      ]
    }

    it "Matrix multiplication" do
      compute = Cl::Brains::CLMatrixMath.new(3)
      puts Benchmark.measure {
        
        expect(compute.prepare(matrix_a, matrix_b).mul_int).to eq([
          [1,	18,	11],
          [2,	-10, -2],
          [-7,	6,	6]
        ])
      }
      puts Benchmark.measure {
      expect((Matrix[*matrix_a] *  Matrix[*matrix_b]).to_a).to eq([
        [1,	18,	11],
        [2,	-10, -2],
        [-7,	6,	6]
      ])
      }
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
        expect(compute.prepare(ma, mb).mul_int).to eq(expected)
      }
    end
  end
end
