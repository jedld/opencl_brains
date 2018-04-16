require "tensor_stream/evaluator/operation_helpers/random_gaussian"

module TensorStream
  ## PURE ruby evaluator used for testing and development
  class RubyEvaluator
    def initialize(session, context)
      @session = session
      @context = context
    end

    def eval(tensor)
      if tensor.kind_of?(Operation)
        eval_operation(tensor)
      elsif tensor.kind_of?(Variable)
        eval_variable(tensor)
      else
        eval_tensor(tensor)
      end
    end
    
    protected

    def eval_variable(tensor)
      raise "variable #{tensor.name} not initalized" if tensor.value.nil?

      eval_tensor(tensor.value)
    end

    def eval_operation(tensor)
      return @context[tensor.name.to_sym] if @context.has_key?(tensor.name.to_sym)

      a = resolve_placeholder(tensor.items[0]) if tensor.items
      b = resolve_placeholder(tensor.items[1]) if tensor.items

      case(tensor.operation)
        when :add
          process_vector_math_op(a, b, ->(a,b) { a + b })
        when :sub
          process_vector_math_op(a, b, ->(a,b) { a - b })
        when :mul
          process_vector_math_op(a, b, ->(a,b) { a * b })
        when :exp
          process_vector_math_op(a, b, ->(a,b) { a ** b })
        when :random_uniform
          maxval = tensor.options.fetch(:maxval, 1)
          minval = tensor.options.fetch(:minval, 0)
  
          generator = ->() { rand * (maxval - minval) + minval } 
          TensorStream.constant(generate_vector(tensor.options[:shape], generator: generator ))
        when :random_normal
          r = RandomGaussian.new(tensor.options.fetch(:mean), tensor.options.fetch(:stddev))
          generator = ->() { r.rand }
          
          TensorStream.constant(generate_vector(tensor.options[:shape], generator: generator ))
        when :flow_group
          tensor.items.each do |item| eval(item) end
          nil
        when :assign
          assign = tensor.items[0] || tensor
          assign.value = eval(tensor.items[1])
          assign.value
        when :assign_add
          tensor.items[0].value = eval(tensor.items[0].value + eval(tensor.items[1]))
          tensor.items[0].value
        when :reduce_sum
          val = eval(tensor.items[0])
          axis = tensor.options[:axis]
          keep_dims = tensor.options[:keepdims]
          res = if axis.kind_of?(Array)
            axis.each do |x|
              val = reduce_axis(x, val, keep_dims)
            end

            val.flatten.reduce(:+)
          else
            reduce_axis(axis, val, keep_dims)
          end
          TensorStream.constant(res)
        when :zeros
          if tensor.shape.shape.size == 0
            TensorStream.constant(0)
          else
            TensorStream.constant(generate_vector(tensor.shape.shape, generator: ->() { 0.0 } ))
          end
        when :matmul
          matrix_a = eval(a)
          matrix_b = eval(b)
  
          TensorStream.constant((Matrix[*matrix_a] *  Matrix[*matrix_b]).to_a)
        when :gradient_descent
        when :div
          process_vector_math_op(a, b, ->(a,b) { a/b })
        else
          raise "unknown op #{tensor.operation}"
      end.tap do |result|
        @context[tensor.name.to_sym] = result
      end
    end

    def eval_tensor(tensor)
      return tensor unless tensor.kind_of?(Tensor)
      return @context[tensor.name] if @context.has_key?(tensor.name)

      if tensor.value.kind_of?(Array)
        tensor.value.collect do |item|
          item.kind_of?(Tensor) ? eval(item) : item
        end
      else
        tensor.value.kind_of?(Tensor) ? eval(tensor.value) : tensor.value
      end.tap do |result|
        @context[tensor.name] = result
      end
    end

    private

    def process_vector_math_op(a, b, op)
      # ruby scalar
      if a.shape.rank == 0
        TensorStream.constant(op.(eval(a),eval(b)), dtype: a.dtype)
      elsif a.shape.rank > 0
        if b.kind_of?(Tensor) && b.shape.rank > 0
          TensorStream.constant(vector_op(a, b, op))
        else
          val = b.kind_of?(Tensor) ? b.value : b
          TensorStream.constant(constant_op(a, val, op))
        end
      end
    end

    def resolve_placeholder(placeholder)
      var = if placeholder.kind_of?(Placeholder) 
        @context[placeholder.name.to_sym].tap do |c|
          raise "missing placeholder #{placeholder.name}" if c.nil?
        end
      else
        placeholder
      end

      var.kind_of?(Operation) ? eval(var) : var
    end

    def reduce_axis(axis, val,  keep_dims, op = ->(v) { v.kind_of?(Array) ? v.reduce(:+) : v })
      val = eval(val)
      res = if axis.nil?
        op.(val.flatten)
      elsif axis == 0
        val.transpose.collect do |v|
          keep_dims ? [op.(v)] : op.(v)
        end
      elsif axis == 1
        val.collect do |v|
          keep_dims ? [op.(v)] : op.(v)
        end
      else
        fail "can't handle with axis > 1 :("
      end
    end

    def constant_add(vector, constant)
      eval(vector).collect do |item|
        if item.is_a?(Array)
          constant_add(item, constant)
        else
          if item.respond_to?(:value) 
            item.value + constant
          else
            item + constant
          end
        end
      end
    end

    def constant_op(vector, constant, op = ->(a,b) { a + b })
      eval(vector).collect do |item|
        if item.is_a?(Array)
          constant_op(item, constant, op)
        else
          if item.respond_to?(:value) 
            op.(item.value, constant)
          else
            op.(item, constant)
          end
        end
      end
    end

    def vector_op(vector, vector2, op = ->(a,b) { a + b })
      v_a = eval(vector)
      v_b = eval(vector2)
    
      v_a.each_with_index.collect do |item, index|
        if item.is_a?(Array)
          constant_op(item, constant, op)
        else
          if item.respond_to?(:value) 
            op.(item.value, v_b[index].value, op)
          else
            op.(item, v_b[index])
          end
        end
      end
    end

    def vector_add(vector, vector2)
      v_a = eval(vector)
      v_b = eval(vector2)
      
      v_a.each_with_index.collect do |item, index|
        if item.is_a?(Array)
          constant_add(item, constant)
        else
          if item.respond_to?(:value) 
            item.value + v_b[index].value
          else
            item + v_b[index]
          end
        end
      end
    end

    def generate_vector(shape, dtype: :float32, generator: )
      if shape.is_a?(Integer)
        shape.times.collect do
          generator.()
        end
      elsif shape.size > 1
        shape[0].times.collect do
          generate_vector(shape[1..shape.size], generator: generator, dtype: dtype)
        end
      elsif shape.size == 1
        shape[0].times.collect do
          generator.()
        end
      elsif shape.size == 0
        generator.()
      end
    end
  end
end