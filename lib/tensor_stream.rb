require "tensor_stream/version"
require 'opencl_ruby_ffi'
require 'narray_ffi'
require 'tensor_stream/types'
require 'tensor_stream/graph'
require 'tensor_stream/session'
require 'tensor_stream/tensor_shape'
require 'tensor_stream/tensor'
require 'tensor_stream/operation'
require "tensor_stream/gemm/gemm"
require "tensor_stream/sigmoid/sigmoid"

module TensorStream
  def self.get_default_graph
    TensorStream::Graph.get_default_graph
  end

  def self.Variable(value, dtype = nil)
    if value.is_a?(String)
      TensorStream::Tensor.new(dtype || :string_ref, 0, [], value: value)
    elsif value.is_a?(Integer)
      TensorStream::Tensor.new(dtype || :int32_ref, 0, [], value: value)
    elsif value.is_a?(Float)
      TensorStream::Tensor.new(dtype || :float32_ref, 0, [], value: value)
    end
  end

  def self.Session
    TensorStream::Session.new
  end

  def self.constant(value, options = {})
    shared_options = { const: true, value: value, name: options[:name] }
    if value.is_a?(Float)
      TensorStream::Tensor.new(options[:dtype] || :float32, 0, [], shared_options)
    elsif value.is_a?(Integer)
      TensorStream::Tensor.new(options[:dtype] || :int32, 0, [], shared_options)
    elsif value.is_a?(String)
      TensorStream::Tensor.new(options[:dtype] || :string, 0, [], shared_options)
    elsif value.is_a?(Array)
      dtype = nil
      rank = 1
      dimensions = []
      value_ptr = value
      begin
        dtype, rank, value_ptr, d = dtype_eval(dtype, rank, value_ptr)
        dimensions << d
      end while dtype == :array

      TensorStream::Tensor.new(dtype, rank, dimensions, shared_options)
    end
  end

  private

  def self.dtype_eval(dtype, rank, value)
    dtype = if value[0].is_a?(String)
      :string
    elsif value[0].is_a?(Float)
      :float32
    elsif value[0].is_a?(Integer)
      :int32
    elsif value[0].is_a?(Array)
      rank += 1
      :array
    else
      :float32
    end

    [dtype, rank, value[0], value.size]
  end
end
