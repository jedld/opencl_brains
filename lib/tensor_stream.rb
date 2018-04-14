require "tensor_stream/version"
require 'opencl_ruby_ffi'
require 'narray_ffi'
require 'tensor_stream/evaluator/ruby_evaluator'
require 'tensor_stream/graph_keys'
require 'tensor_stream/types'
require 'tensor_stream/graph'
require 'tensor_stream/session'
require 'tensor_stream/tensor_shape'
require 'tensor_stream/tensor'
require 'tensor_stream/variable'
require 'tensor_stream/operation'
require 'tensor_stream/placeholder'
require 'tensor_stream/control_flow'
require "tensor_stream/gemm/gemm"
require "tensor_stream/sigmoid/sigmoid"

module TensorStream
  def self.get_default_graph
    TensorStream::Graph.get_default_graph
  end

  def self.Variable(value, options = {})
    if value.is_a?(String)
      TensorStream::Variale.new(options[:dtype] || :string_ref, 0, [], value: value, name: options[:name])
    elsif value.is_a?(Integer)
      TensorStream::Variable.new(options[:dtype] || :int32_ref, 0, [], value: value, name: options[:name])
    elsif value.is_a?(Float)
      TensorStream::Variable.new(options[:dtype] || :float32_ref, 0, [], value: value, name: options[:name])
    end
  end

  def self.Session
    session = TensorStream::Session.new
    if block_given?
      yield session
    end
    session
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

  def self.group(inputs)
    TensorStream::ControlFlow.new(:group, inputs)
  end

  def self.get_variable(name, options = {})
    TensorStream::Variable.new(options[:dtype] || :float32, nil, options[:shape], name: name, initializer: options[:initializer])
  end

  def self.get_collection(name, options = {})
    Graph.get_default_graph.get_collection(name, options)
  end

  def self.placeholder(dtype)
    TensorStream::Placeholder.new(dtype, nil, nil)
  end

  def self.random_uniform(shape: , dtype: :float32)
    options = {shape: shape, dtype: dtype}
    TensorStream::Operation.new(:random_uniform, nil, nil, options)
  end

  def self.global_variables_initializer
    TensorStream::Variable.global_variables_initializer
  end

  def self.zeros_initializer(options = {})
    TensorStream::Operation.new(:zeros, nil, nil, options)
  end

  def self.add(a, b)
    a + b
  end

  def self.multiply(a, b)
    a * b
  end

  def self.matmul(a, b, options = {})
    TensorStream::Operation.new(:matmul, a, b, options)
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
