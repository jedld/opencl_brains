module TensorStream
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank, :options

    def initialize(operation, a, b, options = {})
      @operation = operation
      @items = [a, b]
      @rank = options[:rank] || 0
      @name = set_name
      @graph = options[:graph] || TensorStream.get_default_graph
      @options = options
      if options[:shape]
        @shape = TensorShape.new(options[:shape], options[:shape].size || 0)
      end
      @graph.add_node(self)
    end

    def to_s
      @name
    end

    def self.reset_counters
      @@op_counter = 0
    end

    def self.derivative(tensor, with_respect: [], stop_gradients: [])
      TensorStream.constant(0) if stop_gradients.include?(tensor)

      if tensor.kind_of?(Operation)
        case tensor.operation
          when :sin
            Operation.new(:cos, tensor.items[0], nil) * derivative(tensor.items[0])
          when :cos
            -Operation.new(:sin, tensor.items[0], nil) * derivative(tensor.items[0])
          when :add
            derivative(tensor.items[0]) + derivative(tensor.items[1])
          when :sub
            derivative(tensor.items[0]) - derivative(tensor.items[1])
          when :exp
            tensor.items[1] * (tensor.items[0] ** (tensor.items[1] - 1)) * derivative(tensor.items[0])
          when :div
            # apply the quotient rule
            ( derivative(tensor.items[0]) * tensor.items[1] - tensor.items[0] * derivative(tensor.items[1]) ) / tensor.items[1]**2
          when :mul
            # apply the product rule
            derivative(tensor.items[0]) * tensor.items[1] + tensor.items[0] * derivative(tensor.items[1])
          when :reduce_sum
            derivative(tensor.items[0])
          else
            fail "no derivative found for #{tensor.operation}"
        end
      elsif tensor.kind_of?(TensorStream::Variable) || tensor.kind_of?(TensorStream::Placeholder)
        TensorStream.constant(1,  dtype: tensor.data_type)
      else
        TensorStream.constant(0)
      end
    end

    def to_h
      {
        op: operation,
        name: name,
        operands: hashify_tensor(items)
      }
    end

    def to_math
      case operation
      when :sin
        "sin(#{auto_math(items[0])})"
      when :cos
        "cos(#{auto_math(items[0])})"
      when :add
       "(#{auto_math(items[0])} + #{auto_math(items[1])})"
      when :sub
        "(#{auto_math(items[0])} - #{auto_math(items[1])})"
      when :exp
        "(#{auto_math(items[0])}^#{auto_math(items[1])})"
      when :div
        "(#{auto_math(items[0])} / #{auto_math(items[1])})"
      when :mul
        if auto_math(items[0]) == 1
          auto_math(items[1])
        elsif auto_math(items[1]) == 1
          auto_math(items[0])
        else
          "(#{auto_math(items[0])} * #{auto_math(items[1])})"
        end
      when :reduce_sum
        "reduce_sum(|#{auto_math(items[0])} * #{auto_math(items[1])}|)"
      else
        fail "math form for #{operation}"
      end
    end

    private

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