module TensorStream
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank, :options

    def initialize(operation, a, b, options = {})
      @operation = operation
      @rank = options[:rank] || 0
      @name = options[:name] || set_name
      @graph = options[:graph] || TensorStream.get_default_graph
      @options = options
      @data_type = options[:data_type]

      @items = [a, b].map { |i| options[:preserve_params_type] ? i : auto_wrap(i) } 

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
      constant_options = { dtype: options[:dtype] || tensor.data_type, shape: options[:shape] || (tensor.shape ? tensor.shape.shape : nil)}
      return TensorStream.constant(1, constant_options) if tensor == dx
      return TensorStream.constant(0, constant_options) if options[:stop_gradients] && options[:stop_gradients].include?(tensor)
  
      if tensor.kind_of?(Operation)
        case tensor.operation
          when :abs
            derivative(tensor.items[0], dx, options) * Operation.new(:sign, tensor.items[0], nil)
          when :exp
            Operation.new(:exp, tensor.items[0], nil)
          when :log
            TensorStream.constant(1, constant_options) / tensor.items[0]
          when :stop_gradient
            return TensorStream.constant(0, constant_options)
          when :tanh
            TensorStream.constant(1, constant_options) - (Operation.new(:tanh, tensor.items[0], nil) ** 2)
          when :tan
            TensorStream.constant(1, constant_options) / (Operation.new(:cos, tensor.items[0], nil) ** 2)
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
          when :matmul
            derivative_a = derivative(tensor.items[0], dx, shape: tensor.items[1].shape.shape)
            derivative_b = derivative(tensor.items[1], dx, shape: tensor.items[0].shape.shape)

            Operation.new(:matmul, derivative_a,  tensor.items[1], transpose_b: true,
                name: "matrix_dx" ) +
            Operation.new(:matmul, tensor.items[0], derivative_b, transpose_a: true,
                name: "matrix_dy" )
          else
            fail "no derivative implementation found for op #{tensor.operation}"
        end
      elsif tensor.kind_of?(TensorStream::Variable)
        if tensor == dx
          TensorStream.constant(1, constant_options)
        else
          TensorStream.constant(0, constant_options)
        end
      else
        TensorStream.constant(0, constant_options)
      end
    end

    def to_h
      {
        op: operation,
        name: name,
        operands: hashify_tensor(items)
      }
    end

    def self.empty_matrix?(m)
      if m.kind_of?(Array)
        m.each do |item|
          if item.kind_of?(Array)
            return false if !empty_matrix?(item)
          else
            return false if item!=0 || item!=0.0
          end
        end
      end

      return true
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
      when :sin, :cos, :tanh
        "#{operation}(#{auto_math(items[0])})"
      when :add
       "(#{auto_math(items[0])} + #{auto_math(items[1])})"
      when :sub
        "(#{auto_math(items[0])} - #{auto_math(items[1])})"
      when :pow
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
      when :stop_gradient
        auto_math(items[0])
      when :matmul
        "#{auto_math(items[0])}.matmul(#{auto_math(items[1])})"
      when :eye
        "eye(#{auto_math(items[0])})"
      when :transpose
        "transpose(#{auto_math(items[0])})"
      when :shape
        "#{auto_math(items[0])}.shape"
      when :exp
        "e^#{auto_math(items[0])}"
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