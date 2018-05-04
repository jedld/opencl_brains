require "tensor_stream/evaluator/operation_helpers/random_gaussian"
require 'tensor_stream/math_gradients'

module TensorStream
  class FullEvalNotPossible < Exception
  end

  class EvaluatorExcecutionException < Exception
    attr_reader :tensor

    def initialize(exception, tensor)
      @exception = exception
    end

    def wrapped_exception
      @exception
    end
  end

  ## PURE ruby evaluator used for testing and development
  class RubyEvaluator
    attr_accessor :retain

    include TensorStream::OpHelper

    def initialize(session, context)
      @session = session
      @context = context
      @retain = context[:retain] || []
    end

    def run(tensor, execution_context)
      return tensor.map { |t| run(t, execution_context) } if tensor.is_a?(Array)

      return tensor if retain.include?(tensor) # if var is in retain don't eval to value

      child_context = execution_context.dup
      res = if tensor.is_a?(Operation)
              eval_operation(tensor, child_context)
            elsif tensor.is_a?(Variable)
              eval_variable(tensor, child_context)
            elsif tensor.is_a?(Placeholder)
              resolve_placeholder(tensor, child_context)
            else
              eval_tensor(tensor, child_context)
            end
      execution_context.deep_merge!(returns: child_context[:returns])
      res
    end

    def complete_eval(tensor, context)
      Kernel.loop do
        old_tensor = tensor
        tensor = run(tensor, context)

        if tensor.is_a?(Array) && tensor.size > 0 && tensor[0].is_a?(Tensor)
          tensor = tensor.map { |t| complete_eval(t, context) }
        end

        return tensor if old_tensor.equal?(tensor)
        return tensor unless tensor.is_a?(Tensor)
      end
    end

    protected

    def eval_variable(tensor, child_context)
      fail "variable #{tensor.name} not initalized" if tensor.value.nil?
      eval_tensor(tensor.value, child_context).tap do |val|
        child_context[:returns] ||= {}
        child_context[:returns][:vars] ||= []
        child_context[:returns][:vars] << { name: tensor.name, val: val }
      end
    end

    def eval_operation(tensor, child_context)
      return @context[tensor.name.to_sym] if @context.key?(tensor.name.to_sym)

      a = resolve_placeholder(tensor.items[0], child_context) if tensor.items && tensor.items[0]
      b = resolve_placeholder(tensor.items[1], child_context) if tensor.items && tensor.items[1]

      case tensor.operation
      when :sign
        a = complete_eval(a, child_context)

        func = lambda { |x, _b|
          if x == 0 || (x.is_a?(Float) && x.nan?)
            0
          elsif x < 0
            -1
          elsif x > 0
            1
          else
            fail 'assert: cannot be here'
          end
        }

        call_op(:sign, a, child_context, func)
      when :equal
        a = complete_eval(a, child_context)
        b = complete_eval(b, child_context)

        (a == b)
      when :index
        f = run(a, child_context)
        index = run(b, child_context)

        f[index]
      when :slice
        input = complete_eval(a, child_context)
        start = complete_eval(b, child_context)
        size = complete_eval(tensor.options[:size], child_context)
        fail "start index and size not of the same shape #{start.size} != #{size.size}" if start.size != size.size
        slice_tensor(input, start, size)
      when :negate
        call_vector_op(:negate, a, nil, child_context, ->(t, _u) { -t })
      when :add
        call_vector_op(:add, a, b, child_context, ->(t, u) { t + u })
      when :sub
        call_vector_op(:sub, a, b, child_context, ->(t, u) { t - u })
      when :mul
        call_vector_op(:mul, a, b, child_context, ->(t, u) { t * u })
      when :pow
        call_vector_op(:pow, a, b, child_context, ->(t, u) { t**u })
      when :concat
        values = complete_eval(a, child_context)
        res = concat_array(values, tensor.options[:axis])
        cons(res)
      when :abs
        call_op(:abs, a, child_context, ->(t, _b) { t.abs })
      when :tanh
        call_op(:tanh, a, child_context, ->(t, _b) { Math.tanh(t) })
      when :tan
        call_op(:tan, a, child_context, ->(t, _b) { Math.tan(t) })
      when :sec
        call_op(:sec, a, child_context, ->(t, _b) { Math.sec(t) })
      when :sin
        call_op(:sin, a, child_context, ->(t, _b) { Math.sin(t) })
      when :cos
        call_op(:cos, a, child_context, ->(t, _b) { Math.cos(t) })
      when :log
        call_op(:log, a, child_context, ->(t, _b) { Math.log(t) } )
      when :exp
        call_op(:exp, a, child_context, ->(t, _b) { Math.exp(t) } )
      when :square
        call_op(:square, a, child_context, ->(t, _b) { t * t } )
      when :stop_gradient
        run(a, child_context)
      when :random_uniform
        maxval = tensor.options.fetch(:maxval, 1)
        minval = tensor.options.fetch(:minval, 0)

        generator = ->() { rand * (maxval - minval) + minval }
        cons(generate_vector(tensor.options[:shape], generator: generator ))
      when :random_normal
        r = RandomGaussian.new(tensor.options.fetch(:mean), tensor.options.fetch(:stddev))
        generator = ->() { r.rand }

        cons(generate_vector(tensor.options[:shape], generator: generator ))
      when :flow_group
        tensor.items.each { |item| run(item, child_context) }
        nil
      when :assign
        assign = tensor.items[0] || tensor
        assign.value = run(tensor.items[1], child_context)
        assign.value
      when :assign_add
        tensor.items[0].value = process_vector_math_op(tensor.items[0], tensor.items[1], child_context, ->(a,b) { a + b })
        tensor.items[0].value
      when :assign_sub
        tensor.items[0].value = process_vector_math_op(tensor.items[0], tensor.items[1], child_context, ->(a,b) { a - b })
        tensor.items[0].value
      when :reduce_sum
        val = run(tensor.items[0], child_context)
        axis = tensor.options[:axis]
        keep_dims = tensor.options[:keepdims]
        res = if axis.is_a?(Array)
                axis.each do |x|
                  val = reduce_axis(x, val, keep_dims, child_context)
                end

                val.flatten.reduce(:+)
              else
                reduce_axis(axis, val, keep_dims, child_context)
              end
        cons(res)
      when :transpose
        matrix_a = complete_eval(a, child_context)
        cons(matrix_a.transpose)
      when :eye
        rows = complete_eval(a, child_context)
        columns = complete_eval(b, child_context)

        rows.times.collect do |index|
          columns.times.collect do |col|
            if tensor.data_type == :float32
              index == col ? 1.0 : 0.0
            else
              index == col ? 1 : 0
            end
          end
        end
      when :cond
        pred = complete_eval(tensor.options[:pred], child_context)

        if pred
          complete_eval(a, child_context)
        else
          complete_eval(b, child_context)
        end
      when :less
        a = complete_eval(a, child_context)
        b = complete_eval(b, child_context)

        a < b
      when :greater
        a = complete_eval(a, child_context)
        b = complete_eval(b, child_context)

        a > b
      when :zeros, :ones
        s = complete_eval(a, child_context) || tensor.shape.shape

        func = if tensor.operation == :zeros
                 ->() { tensor.data_type == :int32 ? 0 : 0.0 }
               else
                 ->() { tensor.data_type == :int32 ? 1 : 1.0 }
               end

        if s.is_a?(Array) && s.size == 0
          cons(func.call())
        else
          s = [s.to_i] unless s.is_a?(Array)
          cons(generate_vector(s, generator: func))
        end
      when :shape
        input = complete_eval(a, child_context)

        cons(shape_eval(input))
      when :matmul
        matrix_a = complete_eval(a, child_context)
        matrix_b = complete_eval(b, child_context)

        matrix_a = matrix_a.transpose if tensor.options[:transpose_a]
        matrix_b = matrix_b.transpose if tensor.options[:transpose_b]

        # handle matrix multiplication with constants like 1 or 0
        matrix_a = matmul_const_transform(matrix_a, matrix_b, tensor)
        matrix_b = matmul_const_transform(matrix_b, matrix_a, tensor)

        # check matrix dimensions
        fail "incompatible shape sizes for matrix multiplication (#{matrix_a[0].size} != #{matrix_b.size}) #{shape_eval(matrix_a)} vs #{shape_eval(matrix_b)}" if matrix_a[0].size != matrix_b.size

        cons((Matrix[*matrix_a] * Matrix[*matrix_b]).to_a)
      when :gradients
        b.collect do |xs|
          fail "#{xs} passed is not a tensor object" unless xs.is_a?(Tensor)
          xs_val = complete_eval(xs, child_context)
          target_shape = shape_eval(xs_val)
          derivative = complete_eval(TensorStream::MathGradients.derivative(a, xs, stop_gradients: tensor.options[:stop_gradients], target_shape: target_shape), child_context)

          unit_matrix = cons(generate_vector(target_shape, generator: -> { xs.data_type == :int32 ? 1 : 1.0 } ))
          complete_eval(unit_matrix * cons(derivative), child_context)
        end
      when :identity
        cons(complete_eval(a, child_context))
      when :print
        a = complete_eval(a, child_context)
        b = complete_eval(b, child_context)
        puts "#{tensor.options[:message] || ""} #{b}"
        cons(a)
      when :rank
        a = complete_eval(a, child_context)
        cons(get_rank(a), data_type: :int32)
      when :div
        process_vector_math_op(a, b, child_context, ->(a,b) { a/b })
      when :reshape
        arr = complete_eval(a, child_context)
        new_shape = complete_eval(b, child_context)

        flat_arr = arr.flatten
        return flat_arr[0] if new_shape.size == 0 && flat_arr.size == 1

        new_shape = fix_inferred_elements(new_shape, flat_arr.size)

        cons(reshape(flat_arr, new_shape), dtype: a.data_type)
      else
        fail "unknown op #{tensor.operation}"
      end.tap do |result|
        @context[tensor.name.to_sym] = result
      end
    rescue EvaluatorExcecutionException => e
      raise e
    rescue StandardError => e
      puts e.message
      puts e.backtrace.join("\n")
      raise EvaluatorExcecutionException.new(e, tensor), "error #{e.message} while evaluating #{tensor.name} : #{tensor.to_math}"
    end

    def eval_tensor(tensor, child_context)
      return tensor unless tensor.is_a?(Tensor)
      return @context[tensor.name] if @context.key?(tensor.name)

      if tensor.value.is_a?(Array)
        tensor.value.collect do |item|
          item.is_a?(Tensor) ? run(item, child_context) : item
        end
      else
        tensor.value.is_a?(Tensor) ? run(tensor.value, child_context) : tensor.value
      end.tap do |result|
        @context[tensor.name] = result
      end
    end

    private

    def slice_tensor(input, start, size)
      start_index = start.shift
      dimen_size = start_index + size.shift

      input[start_index...dimen_size].collect do |item|
        if item.is_a?(Array)
          slice_tensor(item, start.dup, size.dup)
        else
          item
        end
      end
    end

    def matmul_const_transform(mat, mat_b, tensor)
      if !mat.is_a?(Array)
        compat_shape = shape_eval(mat_b).reverse
        func = ->() { tensor.data_type == :int32 ? mat.to_i : mat.to_f }

        generate_vector(compat_shape, generator: func)
      else
        mat
      end
    end

    def fix_inferred_elements(shape, total_size)
      return shape if shape.empty?

      current_size = shape.inject(1) { |product, n|  n > 0 ? product * n : product }
      inferred_size = total_size / current_size
      shape.map { |s| s == -1 ? inferred_size : s }
    end

    def reshape(arr, new_shape)
      return arr if new_shape.empty?

      s = new_shape.shift

      if new_shape.size == 0
        fail "reshape dimen mismatch #{arr.size} != #{s}" if arr.size != s
        return arr
      end

      dim = (arr.size / s)
      arr.each_slice(dim).collect do |slice|
        reshape(slice, new_shape.dup)
      end
    end

    def call_op(op, a, child_context, func)
      a = complete_eval(a, child_context)
      process_function_op(a, child_context, func)
    rescue TensorStream::FullEvalNotPossible
      TensorStream.send(op.to_sym, a)
    end

    def call_vector_op(op, a, b, child_context, func)
      process_vector_math_op(a, b,  child_context, func)
    rescue TensorStream::FullEvalNotPossible
      TensorStream.send(op.to_sym, a, b)
    end

    def process_vector_math_op(a, b,  child_context, op)
      eval_a = complete_eval(a, child_context) unless a.nil?
      eval_b = complete_eval(b, child_context) unless b.nil?

      fail FullEvalNotPossible.new, "full eval not possible for #{a.name}" if eval_a.is_a?(Tensor) || eval_b.kind_of?(Tensor)

      # ruby scalar
      if get_rank(eval_a) == 0
        if (get_rank(eval_b)) == 0
          TensorStream.constant(op.call(eval_a,eval_b), dtype: a.dtype)
        else
          TensorStream.constant(constant_op(eval_b, eval_a, child_context, op, true))
        end
      elsif get_rank(eval_a) > 0
        if get_rank(eval_b) > 0
          TensorStream.constant(vector_op(eval_a, eval_b, child_context, op))
        else
          TensorStream.constant(constant_op(eval_a, eval_b, child_context, op))
        end
      end
    end

    def get_rank(value, rank = 0)
      return rank unless value.is_a?(Array)
      return rank + 1 if value.size == 0

      get_rank(value[0], rank + 1)
    end

    def concat_array(values, axis)
      combined_array = values.shift
      axis = get_rank(combined_array) - 1 if axis == -1

      values.each do |v|
        combined_array = concat(combined_array, v, axis)
      end
      combined_array
    end

    def concat(a, b, axis)
      if axis == 0
        a + b
      else
        a.each_with_index.collect do |i, index|
          concat(i, b[index], axis - 1)
        end
      end
    end

    def process_function_op(a, child_context, op)
      # ruby scalar
      if (a.kind_of?(Tensor) && a.shape.rank > 0) || a.kind_of?(Array)
        TensorStream.constant(constant_op(a, 0, child_context, op))
      elsif !a.kind_of?(Tensor) || a.shape.rank == 0
        v = run(a, child_context)
        fail FullEvalNotPossible.new, "full eval not possible for #{v.name}" if v.is_a?(Tensor) && !v.is_const

        TensorStream.constant(op.call(v, 0), dtype: TensorStream.val_to_dtype(v))
      else
        fail 'cannot be here'
      end
    end

    def resolve_placeholder(placeholder, execution_context = {})
      return nil if placeholder.nil?
      return placeholder if retain.include?(placeholder)

      var = if placeholder.kind_of?(Placeholder)
              @context[placeholder.name.to_sym].tap do |c|
                if c.nil?
                  raise "missing placeholder #{placeholder.name}"
                end
              end
            else
              placeholder
            end

      return var unless placeholder.kind_of?(Tensor)
      Tensor.cast_dtype(var, placeholder.data_type)
    end

    def reduce_axis(axis, val,  keep_dims, child_context, op = ->(v) { v.kind_of?(Array) ? v.reduce(:+) : v })
      val = run(val, child_context)
      return val.is_a?(Array) ? op.call(val.flatten) : val if axis.nil?
      return val.transpose.collect { |v| keep_dims ? [op.call(v)] : op.call(v) } if axis == 0
      return val.collect { |v| keep_dims ? [op.call(v)] : op.call(v) } if axis == 1

      fail "can't handle with axis > 1 :("
    end

    def constant_add(vector, constant)
      run(vector).collect do |item|
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

    def constant_op(vector, constant, child_context, op = ->(a,b) { a + b }, switch = false)
      eval_vector = complete_eval(vector, child_context)
      constant = complete_eval(constant, child_context)

      fail FullEvalNotPossible.new, "full eval not possible for #{eval_vector.name}" if eval_vector.kind_of?(Tensor) || constant.kind_of?(Tensor)

      eval_vector.each_with_index.collect do |item, index|
        c = constant.is_a?(Array) ? constant[index] : constant
        if item.is_a?(Array)
          constant_op(item, c, child_context, op, switch)
        else
          if item.respond_to?(:value)
            switch ? op.(c, item.value) : op.(item.value, c)
          else
            switch ? op.(c, item) : op.(item, c)
          end
        end
      end
    end

    def vector_op(vector, vector2, child_context, op = ->(a,b) { a + b })
      v_a = run(vector, child_context)
      v_b = run(vector2, child_context)

      v_a.each_with_index.collect do |item, index|
        if item.is_a?(Array)
          if get_rank(v_a) > get_rank(v_b)
            vector_op(item, v_b, child_context, op) # derank v1
          else
            constant_op(item, v_b[index], child_context, op)
          end
        else
          if item.respond_to?(:value)
            op.(item.value, v_b[index].value)
          else
            op.(item, v_b[index])
          end
        end
      end
    end

    def vector_add(vector, vector2, child_context)
      v_a = run(vector, child_context)
      v_b = run(vector2, child_context)

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
          generator.call
        end
      elsif shape.size > 1
        shape[0].times.collect do
          generate_vector(shape[1..shape.size], generator: generator, dtype: dtype)
        end
      elsif shape.size == 1
        shape[0].times.collect do
          generator.call
        end
      elsif shape.size == 0
        generator.call
      end
    end
  end
end