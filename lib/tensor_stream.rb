require "tensor_stream/version"
require 'opencl_ruby_ffi'
require 'deep_merge'
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
require 'tensor_stream/trainer'
# require 'tensor_stream/libraries/layers'
require "tensor_stream/gemm/gemm"
require "tensor_stream/sigmoid/sigmoid"

module TensorStream
  def self.get_default_graph
    TensorStream::Graph.get_default_graph
  end

  def self.Variable(value, dtype = :float32, options = {})
    common_options= {
      initializer: Operation.new(:assign, nil, value),
      name: options[:name]
    }
    if value.is_a?(String)
      TensorStream::Variable.new(dtype || :string, 0, [], common_options)
    elsif value.is_a?(Integer)
      TensorStream::Variable.new(dtype || :int32, 0, [], common_options)
    elsif value.is_a?(Float)
      TensorStream::Variable.new(dtype || :float32, 0, [], common_options)
    end
  end

  def self.Session
    session = TensorStream::Session.new
    if block_given?
      yield session
    end
    session
  end

  def self.layers
    TensorStream::Layers
  end

  def self.constant(value, options = {})
    shared_options = { const: true, value: value, name: options[:name] }
    if value.is_a?(Float)
      TensorStream::Tensor.new(options[:dtype] || :float32, 0, options[:shape] || [], shared_options)
    elsif value.is_a?(Integer)
      TensorStream::Tensor.new(options[:dtype] || :int32, 0, options[:shape] || [], shared_options)
    elsif value.is_a?(String)
      TensorStream::Tensor.new(options[:dtype] || :string, 0, options[:shape] || [], shared_options)
    elsif value.is_a?(Array)
      dtype = nil
      rank = 1
      dimensions = []
      value_ptr = value
      begin
        dtype, rank, value_ptr, d = dtype_eval(dtype, rank, value_ptr)
        dimensions << d
      end while dtype == :array

      TensorStream::Tensor.new(dtype, rank, options[:shape] || dimensions, shared_options)
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

  def self.placeholder(dtype, options = {})
    TensorStream::Placeholder.new(dtype, nil, options[:shape])
  end

  def self.random_uniform(shape: , dtype: :float32, minval: 0, maxval: 1, seed: nil, name: nil)
    options = {shape: shape, dtype: dtype, minval: minval, maxval: maxval, seed: seed, name: name}
    TensorStream::Operation.new(:random_uniform, nil, nil, options)
  end

  def self.random_normal(shape:, dtype: :float32, mean: 0.0, stddev: 1.0, seed: nil, name: nil)
    options = {shape: shape, dtype: dtype, mean: mean, stddev: stddev, seed: seed, name: name}
    TensorStream::Operation.new(:random_normal, nil, nil, options)
  end

  def self.global_variables_initializer
    TensorStream::Variable.global_variables_initializer
  end

  def self.zeros_initializer(options = {})
    TensorStream::Operation.new(:zeros, nil, nil, options)
  end

  def self.zeros(shape, dtype: :float32, name: nil)
    TensorStream::Operation.new(:zeros, nil, nil, {shape: shape})
  end

  def self.reduce_sum(input_tensor, axis = nil, keepdims: false)
    TensorStream::Operation.new(:reduce_sum, input_tensor, nil, {axis: axis, keepdims: keepdims})
  end

  def self.add(a, b)
    a + b
  end

  def self.multiply(a, b)
    a * b
  end

  def self.pow(a, e)
    a**e
  end

  def self.matmul(a, b, options = {})
    TensorStream::Operation.new(:matmul, a, b, options)
  end

  def self.train
    TensorStream::Trainer
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
