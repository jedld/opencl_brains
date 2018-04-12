module TensorStream
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank

    def initialize(operation, a, b, options = {})
      @operation = operation
      @items = [a, b]
      @rank = options[:rank] || 0
      @name = set_name
      @options = options
      if options[:shape]
        @shape = TensorShape.new(options[:shape], options[:shape].size || 0)
      end
    end
    def to_s
      @name
    end

    def self.reset_counters
      @@op_counter = 0
    end
    def ruby_eval(session = Session.default_session, evaluation_cache = {})
      return evaluation_cache[@name.to_sym] if evaluation_cache.has_key?(@name.to_sym)

      a = resolve_placeholder(@items[0], evaluation_cache)
      b = resolve_placeholder(@items[1], evaluation_cache)

      if operation == :add
        # ruby scalar
        if a.shape.rank == 0
          TensorStream.constant(a.ruby_eval(session, evaluation_cache) + b.ruby_eval(session, evaluation_cache), dtype: a.dtype)
        elsif a.shape.rank > 0
          val = b.kind_of?(Tensor) ? val.value : items[1]
          TensorStream.constant(constant_add(a, val, session, evaluation_cache))
        end
      elsif operation == :mul
        # ruby scalar
        if a.shape.rank == 0
          TensorStream.constant(a.ruby_eval(session, evaluation_cache) * b.ruby_eval(session, evaluation_cache), dtype: a.dtype)
        elsif a.shape.rank == 1
          arr1 = a.ruby_eval(session, evaluation_cache)
          arr2 = b.ruby_eval(session, evaluation_cache)
          arr1.each_with_index.collect do |item, index|
            item * arr2[index]
          end
        end
      elsif operation == :random_uniform
        TensorStream.constant(generate_vector(@options[:shape]))
      end.tap do |result|
        evaluation_cache[@name.to_sym] = result
      end
    end

    private

    def resolve_placeholder(placeholder, evaluation_cache)
      if placeholder.kind_of?(Placeholder) 
        evaluation_cache[placeholder.name.to_sym].tap do |c|
          raise "missing placeholder #{placeholder.name}" if c.nil?
        end
      else
        placeholder
      end
    end

    def constant_add(vector, constant, session, evaluation_cache)
      vector.ruby_eval(session, evaluation_cache).collect do |item|
        if item.is_a?(Array)
          constant_add(item, constant)
        else
          item.value + constant
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

    def self.operation_counter
      @@op_counter ||= 0

      name = if @@op_counter == 0
        ""
      else
        "_#{@@op_counter}"
      end

      @@op_counter += 1
      
      name
    end

    def set_name
      "#{@operation}#{Operation.operation_counter}:#{@rank}"
    end
  end
end