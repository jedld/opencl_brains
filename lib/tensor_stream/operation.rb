module TensorStream
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank, :options

    def initialize(operation, a, b, options = {})
      @operation = operation
      @rank = options[:rank] || 0
      @name = set_name
      @graph = options[:graph] || TensorStream.get_default_graph
      @options = options
      @data_type = options[:data_type]

      @items = [a, b].map { |i| auto_wrap(i) }

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

    def self.derivative(tensor, dx, options = {})
      return TensorStream.constant(1, dtype: tensor.data_type) if tensor == dx
      return TensorStream.constant(0, dtype: tensor.data_type) if options[:stop_gradients] && options[:stop_gradients].include?(tensor)
  
      if tensor.kind_of?(Operation)
        case tensor.operation
          when :tanh
            TensorStream.constant(1, dtype: tensor.data_type) - (Operation.new(:tanh, tensor.items[0], nil) ** 2)
          when :tan
            TensorStream.constant(1, dtype: tensor.data_type) / (Operation.new(:cos, tensor.items[0], nil) ** 2)
          when :sin
            Operation.new(:cos, tensor.items[0], nil) * derivative(tensor.items[0], dx, options)
          when :cos
            -Operation.new(:sin, tensor.items[0], nil) * derivative(tensor.items[0], dx, options)
          when :add
            derivative(tensor.items[0], dx, options) + derivative(tensor.items[1], dx, options)
          when :sub
            derivative(tensor.items[0], dx, options) - derivative(tensor.items[1], dx, options)
          when :pow
            tensor.items[1] * (tensor.items[0] ** (tensor.items[1] - 1)) * derivative(tensor.items[0], dx, options)
          when :div
            # apply the quotient rule
            ( derivative(tensor.items[0], dx, options) * tensor.items[1] - tensor.items[0] * derivative(tensor.items[1], dx, options) ) / tensor.items[1]**2
          when :mul
            # apply the product rule
            derivative(tensor.items[0], dx, options) * tensor.items[1] + tensor.items[0] * derivative(tensor.items[1], dx, options)
          when :reduce_sum
            derivative(tensor.items[0], dx, options)
          else
            fail "no derivative found for #{tensor.operation}"
        end
      elsif tensor.kind_of?(TensorStream::Variable)
        if tensor == dx
          TensorStream.constant(1,  dtype: tensor.data_type)
        else
          TensorStream.constant(0, dtype: tensor.data_type)
        end
      else
        TensorStream.constant(0, dtype: tensor.data_type)
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
      when :slice
        "#{auto_math(items[0])}[#{items[1]}]"
      when :assign_sub
        "(#{items[0] ? items[0].name : "self"} -= #{auto_math(items[1])})"
      when :assign_add
        "(#{items[0] ? items[0].name : "self"} += #{auto_math(items[1])})"
      when :assign
        "(#{items[0] ? items[0].name : "self"} = #{auto_math(items[1])})"
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
      when :gradients
        "gradient(#{auto_math(items[0])})"
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