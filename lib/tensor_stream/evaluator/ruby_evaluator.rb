require "tensor_stream/evaluator/operation_helpers/random_gaussian"

module TensorStream
  class FullEvalNotPossible < StandardError
  end
  ## PURE ruby evaluator used for testing and development
  class RubyEvaluator
    attr_accessor :retain
    def initialize(session, context)
      @session = session
      @context = context
      @retain = context[:retain] || []
    end

    def eval(tensor, execution_context)
      if tensor.kind_of?(Array)
        return tensor.map { |t| eval(t, execution_context) }
      end

      return tensor if retain.include?(tensor) # if var is in retain don't eval to value

      child_context = execution_context.dup
      res = if tensor.kind_of?(Operation)
        eval_operation(tensor, child_context)
      elsif tensor.kind_of?(Variable)
        eval_variable(tensor, child_context)
      elsif tensor.kind_of?(Placeholder)
        resolve_placeholder(tensor, child_context)
      else
        eval_tensor(tensor, child_context)
      end
      execution_context.deep_merge!(returns: child_context[:returns])
      res
    end

    def complete_eval(tensor, context)
      begin
        old_tensor = tensor
        tensor = eval(tensor, context)

        if tensor.kind_of?(Array)
          tensor = tensor.map { |t| complete_eval(t, context) }
        end

        return tensor if old_tensor == tensor
      end while tensor.kind_of?(Tensor)
      tensor
    end
    
    protected

    def eval_variable(tensor, child_context)
      raise "variable #{tensor.name} not initalized" if tensor.value.nil?
      eval_tensor(tensor.value, child_context).tap do |val|
        child_context[:returns] ||= {}
        child_context[:returns][:vars] ||= []
        child_context[:returns][:vars] << { name: tensor.name, val: val }
      end
    end

    def eval_operation(tensor, child_context)
      begin
        return @context[tensor.name.to_sym] if @context.has_key?(tensor.name.to_sym)

        a = resolve_placeholder(tensor.items[0], child_context) if tensor.items
        b = resolve_placeholder(tensor.items[1], child_context) if tensor.items

        case(tensor.operation)
          when :sign
            a = complete_eval(a, child_context)

            func = ->(x,b) {
              if x == 0 || (x.kind_of?(Float) && x.nan?)
                  0
              elsif x < 0
                -1
              elsif x > 0
                1
              else
                fail "cannot be here"
              end
            }

            call_op(:sign, a, child_context, func )
          when :equal
            a = complete_eval(a, child_context)
            b = complete_eval(b, child_context)

            (a == b)
          when :slice
            f = eval(a, child_context)
            index = eval(b, child_context)

            f[index]
          when :negate
            process_vector_math_op(a, nil, child_context, ->(a,b) { -a } )
          when :add
            begin
              process_vector_math_op(a, b, child_context, ->(a,b) { a + b })
            rescue TensorStream::FullEvalNotPossible => e
              a + b
            end
          when :sub
            process_vector_math_op(a, b, child_context, ->(a,b) { a - b })
          when :mul
            begin
              process_vector_math_op(a, b, child_context, ->(a,b) { a * b })
            rescue TensorStream::FullEvalNotPossible => e
              a * b
            end
          when :pow
            begin
              process_vector_math_op(a, b, child_context, ->(a,b) { a ** b })
            rescue TensorStream::FullEvalNotPossible => e
              TensorStream.pow(a,b)
            end
          when :concat
            values = complete_eval(a, child_context)
            res = concat_array(values, tensor.options[:axis])
            TensorStream.constant(res)
          when :abs
            call_op(:abs, a, child_context, ->(a,b) { a.abs })
          when :tanh
            call_op(:tanh, a, child_context, ->(a,b) { Math.tanh(a) })
          when :tan
            call_op(:tan, a, child_context, ->(a,b) { Math.tan(a) })
          when :sec
            call_op(:sec, a, child_context, ->(a,b) { Math.sec(a) })        
          when :sin
            call_op(:sin, a, child_context, ->(a,b) { Math.sin(a) })
          when :cos
            call_op(:cos, a, child_context, ->(a,b) { Math.cos(a) })
          when :log
            call_op(:log, a, child_context, ->(a,b) { Math.log(a) } )
          when :exp
            call_op(:exp, a, child_context, ->(a,b) { Math.exp(a) } )
          when :stop_gradient
            eval(a, child_context)
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
            tensor.items.each do |item| eval(item, child_context) end
            nil
          when :assign
            assign = tensor.items[0] || tensor
            assign.value = eval(tensor.items[1], child_context)
            assign.value
          when :assign_add
            tensor.items[0].value = process_vector_math_op(tensor.items[0], tensor.items[1], child_context, ->(a,b) { a + b })
            tensor.items[0].value
          when :assign_sub
            tensor.items[0].value = process_vector_math_op(tensor.items[0], tensor.items[1], child_context, ->(a,b) { a - b })
            tensor.items[0].value          
          when :reduce_sum
            val = eval(tensor.items[0], child_context)
            axis = tensor.options[:axis]
            keep_dims = tensor.options[:keepdims]
            res = if axis.kind_of?(Array)
              axis.each do |x|
                val = reduce_axis(x, val, keep_dims, child_context)
              end

              val.flatten.reduce(:+)
            else
              reduce_axis(axis, val, keep_dims, child_context)
            end
            TensorStream.constant(res)
          when :transpose
            matrix_a = eval(a, child_context)
            TensorStream.constant(matrix_a.transpose)
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
          when :zeros
            if tensor.shape.shape.size == 0
              TensorStream.constant(0)
            else
              TensorStream.constant(generate_vector(tensor.shape.shape, generator: ->() { 0.0 } ))
            end
          when :shape
            input = complete_eval(a, child_context)

            TensorStream.constant(shape_eval(input))
          when :matmul
            matrix_a = complete_eval(a, child_context)
            matrix_b = complete_eval(b, child_context)

            matrix_a = matrix_a.transpose if tensor.options[:transpose_a]
            matrix_b = matrix_b.transpose if tensor.options[:transpose_b]
            TensorStream.constant((Matrix[*matrix_a] *  Matrix[*matrix_b]).to_a)
          when :gradients
            b.collect do |xs|
              fail "#{xs} passed is not a tensor object" unless xs.kind_of?(Tensor)
              Operation.derivative(a, xs, stop_gradients: tensor.options[:stop_gradients])
            end
          when :div
            process_vector_math_op(a, b, child_context, ->(a,b) { a/b })
          when :reshape
            arr = complete_eval(a, child_context)
            new_shape = complete_eval(tensor.options[:shape], child_context)

            flat_arr = arr.flatten
            return flat_arr[0] if new_shape.size == 0 && flat_arr.size == 1

            new_shape = fix_inferred_elements(new_shape, flat_arr.size)
            TensorStream.constant(reshape(flat_arr, new_shape), dtype: a.data_type)
          else
            raise "unknown op #{tensor.operation}"
        end.tap do |result|
          @context[tensor.name.to_sym] = result
        end
      rescue StandardError => e
        puts "error #{e.message} while evaluating #{tensor.name} : #{tensor.to_math}"
        raise e
      end
    end
    def eval_tensor(tensor, child_context)
      return tensor unless tensor.kind_of?(Tensor)
      return @context[tensor.name] if @context.has_key?(tensor.name)

      if tensor.value.kind_of?(Array)
        tensor.value.collect do |item|
          item.kind_of?(Tensor) ? eval(item, child_context) : item
        end
      else
        tensor.value.kind_of?(Tensor) ? eval(tensor.value, child_context) : tensor.value
      end.tap do |result|
        @context[tensor.name] = result
      end
    end

    private
    
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
      begin
        process_function_op(a, child_context, func )
      rescue TensorStream::FullEvalNotPossible => e
        TensorStream.send(op.to_sym, a)
      end
    end

    def derivative_builder(tensor, learning_rate)
      if tensor.kind_of?(Operation)
        op_val = @context[tensor.name]
        context = {}
        error = complete_eval(tensor, context)
        derivative_function = Operation.derivative(tensor)
      elsif tensor.kind_of?(Variable)
        tensor.assign_sub(error * learning_rate)
      end
    end

    def process_vector_math_op(a, b,  child_context, op)
      eval_a = complete_eval(a, child_context) unless a.nil?
      eval_b = complete_eval(b, child_context) unless b.nil?

      raise FullEvalNotPossible.new if eval_a.kind_of?(Tensor) || eval_b.kind_of?(Tensor)

      # ruby scalar
      if get_rank(eval_a) == 0
        if (get_rank(eval_b)) == 0
          TensorStream.constant(op.(eval_a,eval_b), dtype: a.dtype)
        else
          TensorStream.constant(constant_op(eval_b, eval_a, child_context, op, true))
        end
      elsif get_rank(eval_a) > 0
        if eval_b.kind_of?(Tensor) && get_rank(eval_b) > 0
          TensorStream.constant(vector_op(eval_a, eval_b, child_context, op))
        else
          TensorStream.constant(constant_op(eval_a, eval_b, child_context, op))
        end
      end
    end

    def get_rank(value, rank = 0)
      return rank unless value.kind_of?(Array)
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
      if (axis == 0)
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
        v = eval(a, child_context)
        raise FullEvalNotPossible.new if v.is_a?(Tensor) && !v.is_const

        TensorStream.constant(op.(v, 0), dtype: TensorStream.val_to_dtype(v))
      else
        fail "cannot be here"
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
      val = eval(val, child_context)
      res = if axis.nil?
        val.kind_of?(Array) ? op.(val.flatten) : val
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

    def shape_eval(input)
      return [] unless input.kind_of?(Array)
      arr = []
      arr_ptr = input

      begin
        arr << arr_ptr.size
        arr_ptr = arr_ptr[0]
      end while arr_ptr.kind_of?(Array)

      arr
    end

    def constant_op(vector, constant, child_context, op = ->(a,b) { a + b }, switch = false)
      eval_vector = complete_eval(vector, child_context)
      constant = complete_eval(constant, child_context)

      raise FullEvalNotPossible.new if eval_vector.kind_of?(Tensor) || constant.kind_of?(Tensor)

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
      v_a = eval(vector, child_context)
      v_b = eval(vector2, child_context)
    
      v_a.each_with_index.collect do |item, index|
        if item.is_a?(Array)
          constant_op(item, v_b[index], child_context, op)
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
      v_a = eval(vector, child_context)
      v_b = eval(vector2, child_context)
      
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