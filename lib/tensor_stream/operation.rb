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

    def self.derivative(tensor)
      if tensor.kind_of?(Operation)
        case tensor.operation
          when :sin
            Operation.new(:cos, tensor.items[0])
          when :cos
            Operation.new(:sin, tensor.items[0])
          when :sum
            derivative(tensor.ites[0]) + derivative(tensor.items[1])
          when :sub
            derivative(tensor.items[0]) - derivative(tensor.items[1])
          when :exp
            tensor.items[1] * (tensor.items[0] ** (tensor.items[1] - 1))
          when :div
            # apply the quotient rule
            ( derivative(tensor.items[0]) * tensor.items[1] - tensor.items[0] * derivative(tensor.items[1]) ) / tensor.items[1]**2
          when :mul
            # apply the product rule
            derivative(tensor.items[0]) * tensor.items[1] + tensor.items[0] * derivative(tensor.items[1])
          else
            fail "no derivative found for #{tensor.operation}"
        end
      elsif tensor.kind_of?(Variable)
        if tensor.value.nil?
          derivative(tensor.value)
        else
          Operation.new(:derivative, tensor, nil)
        end
      else
        TensorStream.constant(0, shape: tensor.shape.shape)
      end  
    end

    def to_h
      {
        op: operation,
        name: name,
        operands: hashify_tensor(items)
      }
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