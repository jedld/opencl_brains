module TensorStream
  class RubyEvaluator
    def initialize(session, context)
      @session = session
      @context = context
    end

    def eval(tensor)
      if tensor.kind_of?(Operation)
        eval_operation(tensor)
      else
        eval_tensor(tensor)
      end
    end
    
    protected

    def eval_operation(tensor)
      return @context[tensor.name.to_sym] if @context.has_key?(tensor.name.to_sym)

      a = resolve_placeholder(tensor.items[0])
      b = resolve_placeholder(tensor.items[1])

      if tensor.operation == :add
        # ruby scalar
        if a.shape.rank == 0
          TensorStream.constant(eval(a) + eval(b), dtype: a.dtype)
        elsif a.shape.rank > 0
          if b.kind_of?(Tensor) && b.shape.rank > 0
            TensorStream.constant(vector_add(a, b))
          else
            val = b.kind_of?(Tensor) ? b.value : b
            TensorStream.constant(constant_add(a, val))
          end
        end
      elsif tensor.operation == :mul
        # ruby scalar
        if a.shape.rank == 0
          TensorStream.constant(eval(a) * eval(b), dtype: a.dtype)
        elsif a.shape.rank == 1
          arr1 = eval(a)
          arr2 = eval(b)
          arr1.each_with_index.collect do |item, index|
            item * arr2[index]
          end
        end
      elsif tensor.operation == :random_uniform
        TensorStream.constant(generate_vector(tensor.options[:shape]))
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

    def resolve_placeholder(placeholder)
      if placeholder.kind_of?(Placeholder) 
        @context[placeholder.name.to_sym].tap do |c|
          raise "missing placeholder #{placeholder.name}" if c.nil?
        end
      else
        placeholder
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

    def generate_vector(shape, dtype: :float32)
      if shape.size > 1
        shape[0].times.collect do
          generate_vector(shape[1..shape.size])
        end
      else
        shape[0].times.collect do
          rand
        end
      end
    end
  end
end