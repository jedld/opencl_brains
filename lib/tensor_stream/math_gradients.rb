module TensorStream
  class MathGradients
    extend TensorStream::OpHelper

    def self.derivative(tensor, dx, options = {})
      constant_options = { dtype: options[:dtype] || tensor.data_type,
                           shape: options[:shape] || (tensor.shape ? tensor.shape.shape : nil)}
      return cons(1, constant_options) if tensor == dx
      return cons(0, constant_options) if options[:stop_gradients] && options[:stop_gradients].include?(tensor)

      if tensor.is_a?(Operation)
        case tensor.operation
        when :abs
          derivative(tensor.items[0], dx, options) * op(:sign, tensor.items[0])
        when :exp
          op(:exp, tensor.items[0])
        when :log
          cons(1, constant_options) / _ds(tensor.items[0])
        when :stop_gradient
          return cons(0, constant_options)
        when :tanh
          cons(1, constant_options) - (op(:tanh, tensor.items[0])**2)
        when :tan
          cons(1, constant_options) / (op(:cos, tensor.items[0])**2)
        when :sin
          op(:cos, tensor.items[0]) * derivative(tensor.items[0], dx, options)
        when :cos
          -op(:sin, tensor.items[0]) * derivative(tensor.items[0], dx, options)
        when :add
          derivative(tensor.items[0], dx, options) + derivative(tensor.items[1], dx, options)
        when :sub
          derivative(tensor.items[0], dx, options) - derivative(tensor.items[1], dx, options)
        when :pow
          _ds(tensor.items[1]) * (_ds(tensor.items[0])**(_ds(tensor.items[1]) - 1)) * derivative(tensor.items[0], dx, options)
        when :div
          # apply the quotient rule
          (derivative(tensor.items[0], dx, options) * _ds(tensor.items[1]) - _ds(tensor.items[0]) * derivative(tensor.items[1], dx, options) ) / tensor.items[1]**2
        when :mul
          # apply the product rule
          derivative(tensor.items[0], dx, options) * _ds(tensor.items[1]) + _ds(tensor.items[0]) * derivative(tensor.items[1], dx, options)
        when :reduce_sum
          derivative(tensor.items[0], dx, options)
        when :matmul
          tensor_shape1 = tensor.items[1].shape ? tensor.items[1].shape.shape : nil
          tensor_shape0 = tensor.items[0].shape ? tensor.items[0].shape.shape : nil

          derivative_a = derivative(tensor.items[0], dx, shape: tensor_shape1)
          derivative_b = derivative(tensor.items[1], dx, shape: tensor_shape0)

          # derivative_a = op(:reshape, derivative_a, op(:shape, tensor.items[1]))
          # derivative_b = op(:reshape, derivative_b, op(:shape, tensor.items[0]))
          op(:matmul, derivative_a, tensor.items[1], transpose_b: true,
                                                     name:        'matrix_dx') +
            op(:matmul, tensor.items[0], derivative_b, transpose_a: true,
                                                      name:        'matrix_dy')                                  
        else
          fail "no derivative implementation found for op #{tensor.operation}"
        end
      elsif tensor.is_a?(TensorStream::Variable)
        if tensor == dx
          op(:ones, op(:shape, tensor), data_type: tensor.data_type)
        else
          op(:zeros, op(:shape, tensor), data_type: tensor.data_type)
        end
      elsif tensor.is_a?(TensorStream::Placeholder)
        cons(0, constant_options)
      else
        cons(0, constant_options)
      end
    end

    def self._ds(tensor)
      return tensor unless tensor.is_a?(Operation)

      case tensor.operation
      when :reduce_sum
        tensor.items[0]
      else
        tensor
      end
    end
  end
end