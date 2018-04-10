require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe TensorStream do
  describe ".VERSION" do
    it "returns the version" do
      expect(TensorStream.version).to eq("0.1.0")
    end
  end
end