module TensorStream
  class MathGradients
    extend TensorStream::OpHelper

    def self.derivative(tensor, dx, options = {})
      constant_options = { dtype: options[:dtype] || tensor.data_type }
      target_shape = options[:target_shape]
      return cons(1, constant_options) if tensor.equal?(dx)
      return cons(0, constant_options) if options[:stop_gradients] && _include?(options[:stop_gradients], tensor)

      if tensor.is_a?(Operation)
        grad = derivative(tensor.items[0], dx, options)

        case tensor.operation
        when :identity, :print
          grad
        when :negate
          return cons(0, constant_options) if grad.value == 0

          cons(-1, constant_options) * grad
        when :abs
          return cons(0, constant_options) if grad.value == 0

          grad * op(:sign, tensor.items[0])
        when :square
          return cons(0, constant_options) if grad.value == 0

          cons(2, constant_options) * tensor.items[0] * grad
        when :exp
          return cons(0, constant_options) if grad.value == 0

          op(:exp, tensor.items[0]) * grad
        when :log
          return cons(0, constant_options) if grad.value == 0

          (cons(1, constant_options) / _ds(tensor.items[0])) * grad
        when :tanh
          return cons(0, constant_options) if grad.value == 0

          (cons(1, constant_options) - (op(:tanh, tensor.items[0])**2)) * grad
        when :tan
          return cons(0, constant_options) if grad.value == 0

          (cons(1, constant_options) / (op(:cos, tensor.items[0])**2)) * grad
        when :sin
          return cons(0, constant_options) if grad.value == 0

          op(:cos, tensor.items[0]) * grad
        when :cos
          return cons(0, constant_options) if grad.value == 0

          -op(:sin, tensor.items[0]) * grad
        when :add
          grad + derivative(tensor.items[1], dx, options)
        when :sub
          grad - derivative(tensor.items[1], dx, options)
        when :pow
          return cons(0, constant_options) if grad.value == 0

          _ds(tensor.items[1]) * (_ds(tensor.items[0])**(_ds(tensor.items[1]) - 1)) * grad
        when :div
          # apply the quotient rule
          (grad * _ds(tensor.items[1]) - _ds(tensor.items[0]) * derivative(tensor.items[1], dx, options) ) / tensor.items[1]**2
        when :mul
          # apply the product rule
          grad * _ds(tensor.items[1]) + _ds(tensor.items[0]) * derivative(tensor.items[1], dx, options)
        when :reduce_sum
          grad
        when :stop_gradient
          return cons(0, constant_options)
        when :matmul
          tensor_shape1 = tensor.items[1].shape ? tensor.items[1].shape.shape : nil
          tensor_shape0 = tensor.items[0].shape ? tensor.items[0].shape.shape : nil

          derivative_a = derivative(tensor.items[0], dx, shape: tensor_shape1)
          derivative_b = derivative(tensor.items[1], dx, shape: tensor_shape0)

          # derivative_a = op(:reshape, derivative_a, op(:shape, tensor.items[1]))
          # derivative_b = op(:reshape, derivative_b, op(:shape, tensor.items[0]))
          matmul_da = op(:matmul, derivative_a, tensor.items[1], transpose_b: true,
                                                     name:        'matrix_dx')
          matmul_db = op(:matmul, tensor.items[0], derivative_b, transpose_a: true,
                                                     name:        'matrix_dy')
          begin_a = op(:zeros, op(:rank, tensor.items[0]), nil, data_type: :int32)
          begin_b = op(:zeros, op(:rank, tensor.items[1]), nil, data_type: :int32)

          end_a = op(:shape, tensor.items[0])
          end_b = op(:shape, tensor.items[1])
          norm_a = op(:slice, matmul_da, begin_a, size: end_a)
          norm_b = op(:slice, matmul_db, begin_b, size: end_b)
          zero_vect = op(:zeros, target_shape)

          op(:cond, norm_a, zero_vect, pred: op(:shape, norm_a) == target_shape) + op(:cond, norm_b, zero_vect, pred: op(:shape, norm_b) == target_shape)
        else
          fail "no derivative implementation found for op #{tensor.operation}"
        end
      elsif tensor.is_a?(TensorStream::Variable)
        if tensor.equal?(dx)
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

    private

    def self._include?(arr, obj)
      arr.each { |a| return true if a.equal?(obj) }
      false
    end
  end
end