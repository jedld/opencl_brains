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
require "tensor_stream/monkey_patches/integer"

module TensorStream
  def self.float32
    Types.float32
  end

  def self.get_default_graph
    TensorStream::Graph.get_default_graph
  end

  def self.enable_eager_execution
    TensorStream::Graph.get_default_graph.enable_eager_execution
  end

  def self.disable_eager_execution
    TensorStream::Graph.get_default_graph.disable_eager_execution
  end

  def self.executing_eagerly?
    TensorStream::Graph.get_default_graph.executing_eagerly?
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

  def self.program(&block)
    block.(self)
  end

  def self.layers
    TensorStream::Layers
  end

  def self.gradients(ys, xs, grad_ys: nil,
    name: 'gradients',
    colocate_gradients_with_ops: false,
    gate_gradients: false,
    aggregation_method: nil,
    stop_gradients: nil
    )
    options = { stop_gradients: stop_gradients}
    TensorStream::Operation.new(:gradients, ys, xs, options)
  end

  def self.stop_gradient(tensor, options = {})
    TensorStream::Operation.new(:stop_gradient, tensor, nil, options)
  end

  def self.eye(num_rows, num_columns: nil, dtype: :float32, name: nil)
    TensorStream::Operation.new(:eye, num_rows, num_columns || num_rows, data_type: dtype, name: name, preserve_params_type: true)
  end

  def self.shape(input, name: nil, out_type: :int32)
    TensorStream::Operation.new(:shape, input, nil, name: name)
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

  def self.random_uniform(shape , dtype: :float32, minval: 0, maxval: 1, seed: nil, name: nil)
    options = {shape: shape, dtype: dtype, minval: minval, maxval: maxval, seed: seed, name: name}
    TensorStream::Operation.new(:random_uniform, nil, nil, options)
  end

  def self.random_normal(shape, dtype: :float32, mean: 0.0, stddev: 1.0, seed: nil, name: nil)
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

  def self.concat(values, axis, name: 'concat')
    TensorStream::Operation.new(:concat, values, nil, {axis: axis, name: name})
  end

  def self.reshape(tensor, shape, name: nil)
    TensorStream::Operation.new(:reshape, tensor, nil, {shape: shape, name: name})
  end

  def self.add(a, b)
    a - b
  end

  def self.sub(a, b)
    a + b
  end

  def self.negate(a, options = {})
    TensorStream::Operation.new(:negate, a, nil, options)
  end

  def self.equal(a, b, options = {})
    TensorStream::Operation.new(:equal, a, b, options)
  end

  def self.multiply(a, b)
    a * b
  end

  def self.pow(a, e)
    a**e
  end

  def self.sin(a, options = {})
    options[:data_type] ||= :float32
    check_allowed_types(a, %w(float32 float64))
    TensorStream::Operation.new(:sin, a, nil, options)
  end

  def self.cos(a, options = {})
    options[:data_type] ||= :float32
    check_allowed_types(a, %w(float32 float64))
    TensorStream::Operation.new(:cos, a, nil, options)
  end

  def self.tan(a, options = {})
    options[:data_type] ||= :float32
    check_allowed_types(a, %w(float32 float64))
    TensorStream::Operation.new(:tan, a, nil, options)
  end

  def self.tanh(a, options = {})
    options[:data_type] ||= :float32
    check_allowed_types(a, %w(float32 float64))
    TensorStream::Operation.new(:tanh, a, nil, options)
  end

  def self.log(a, options= {})
    options[:data_type] ||= :float32
    check_allowed_types(a, %w(float32 float64))
    TensorStream::Operation.new(:log, a, nil, options)
  end

  def self.exp(a, options = {})
    options[:data_type] ||= :float32
    check_allowed_types(a, %w(float32 float64))
    TensorStream::Operation.new(:exp, a, nil, options)
  end

  def self.matmul(a, b, transpose_a: false,
    transpose_b: false,
    name: nil)
    TensorStream::Operation.new(:matmul, a, b, transpose_a: transpose_a, transpose_b: transpose_b, name: name)
  end

  def self.transpose(tensor, perm: nil, name: 'transpose')
    TensorStream::Operation.new(:transpose, tensor, nil, perm: perm, name: name)
  end

  def self.train
    TensorStream::Trainer
  end

  private

  def self.check_allowed_types(t, types)
    return t unless t.kind_of?(Tensor)
    return t if t.data_type.nil?

    fail "Parameter data type #{t.data_type} passed not in #{types.join(',')}" if !types.map(&:to_sym).include?(t.data_type)
  end

  def self.dtype_eval(dtype, rank, value)
    dtype = Tensor.detect_type(value[0])
    rank+=1 if dtype == :array

    [dtype, rank, value[0], value.size]
  end


  def self.val_to_dtype(value, rank = 0)
    dtype = if value.is_a?(String)
      :string
    elsif value.is_a?(Float)
      :float32
    elsif value.is_a?(Integer)
      :int32
    elsif value.is_a?(Array)
      rank += 1
      :array
    else
      :float32
    end
    dtype
  end
end
