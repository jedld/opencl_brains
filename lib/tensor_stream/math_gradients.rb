module TensorStream
  class MathGradients
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
            TensorStream.constant(1, constant_options) / _ds(tensor.items[0])
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
            _ds(tensor.items[1]) * (_ds(tensor.items[0]) ** (_ds(tensor.items[1]) - 1)) * derivative(tensor.items[0], dx, options)
          when :div
            # apply the quotient rule
            ( derivative(tensor.items[0], dx, options) * _ds(tensor.items[1]) - _ds(tensor.items[0]) * derivative(tensor.items[1], dx, options) ) / tensor.items[1]**2
          when :mul
            # apply the product rule
            derivative(tensor.items[0], dx, options) * _ds(tensor.items[1]) + _ds(tensor.items[0]) * derivative(tensor.items[1], dx, options)
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

    def self._ds(tensor)
      return tensor unless tensor.kind_of?(Operation)

      case tensor.operation
      when :reduce_sum
        tensor.items[0]
      else
        tensor
      end
    end
  end
end