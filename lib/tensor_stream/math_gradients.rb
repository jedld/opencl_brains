module TensorStream
  # Class that provides auto-differentiation
  class MathGradients
    extend TensorStream::OpHelper

    def self.derivative(tensor, dx, options = {})
      gradient_program_name = "_grad_#{tensor.name}_#{dx.name}"
      return options[:graph].get_node(gradient_program_name) if options[:graph] && options[:graph].node_added?(gradient_program_name)

      constant_options = { dtype: options[:dtype] || tensor.data_type}
      target_shape = options[:target_shape]
      return cons(1, constant_options) if tensor.equal?(dx)
      return cons(0, constant_options) if options[:stop_gradients] && _include?(options[:stop_gradients], tensor)

      if tensor.is_a?(Operation)
        grad = derivative(tensor.items[0], dx, options) if tensor.items[0]
        grad2 = derivative(tensor.items[1], dx, options) if tensor.items[1]

        case tensor.operation
        when :where
          x_mask = op(:where, op(:ones_like, tensor.items[0]), op(:zeros_like, tensor.items[1]), pred: tensor.options[:pred])
          y_mask = op(:where, op(:zeros_like, tensor.items[0]), op(:ones_like, tensor.items[1]), pred: tensor.options[:pred])
          x_mask * grad + y_mask * grad2
        when :cond
          op(:cond, grad, grad2, pred: tensor.options[:pred])
        when :identity, :print, :pad
          grad
        when :negate
          cons(-1, constant_options) * grad
        when :abs
          grad * op(:sign, _ds(tensor.items[0]))
        when :square
          cons(2, constant_options) * _ds(tensor.items[0]) * grad
        when :exp
          op(:exp, tensor.items[0]) * grad
        when :log
          (cons(1, constant_options) / _ds(tensor.items[0])) * grad
        when :tanh
          (cons(1, constant_options) - (op(:tanh, _ds(tensor.items[0]))**2)) * grad
        when :tan
          (cons(1, constant_options) / (op(:cos, _ds(tensor.items[0]))**2)) * grad
        when :sin
          op(:cos, tensor.items[0]) * grad
        when :sqrt
          cons(1, constant_options) / (cons(2, constant_options) * op(:sqrt, _ds(tensor.items[0]))) * grad
        when :cos
          -op(:sin, tensor.items[0]) * grad
        when :add
          grad_with_broadcast(tensor, dx, ->(a,b) { op(:add, a, b, name: 'grad_sum') } , options)
        when :sub
          grad_with_broadcast(tensor, dx, ->(a,b) { op(:sub, a, b, name: 'grad_sub') } , options)
        when :pow
          gx = _ds(tensor.items[1])*( _ds(tensor.items[0])**(_ds(tensor.items[1]) - 1)) * grad

          log_x = op(:where, op(:log, tensor.items[0], nil, name: 'log_pow_grad'), op(:zeros_like, tensor.items[0]), pred: tensor.items[0] > 0)
          gy = _ds(tensor.items[0])**_ds(tensor.items[1]) * log_x * grad2

          gx + gy
        when :div
          # apply the quotient rule
          gx = op(:div, grad, _ds(tensor.items[1]))
          gy = grad2 * op(:div, op(:div, -_ds(tensor.items[0]), _ds(tensor.items[1])), _ds(tensor.items[1]))

          gx + gy
        when :mul
          # apply the product rule
          grad * _ds(tensor.items[1]) + _ds(tensor.items[0]) * grad2
        when :reduce_sum
          grad
        when :stop_gradient
          return cons(0, constant_options)
        when :matmul
          tensor_shape1 = tensor.items[1].shape ? tensor.items[1].shape.shape : nil
          tensor_shape0 = tensor.items[0].shape ? tensor.items[0].shape.shape : nil

          derivative_a = derivative(tensor.items[0], dx, target_shape: target_shape)
          derivative_b = derivative(tensor.items[1], dx, target_shape: target_shape)

          s0 =  op(:shape, tensor.items[0])
          s1 =  op(:shape, tensor.items[1])

          identity_0 = op(:ones, [s0[0], s1[1]], nil, data_type: tensor.items[0].data_type)
          identity_1 = op(:ones, [s0[0], s1[1]], nil, data_type: tensor.items[1].data_type)

          matmul_da = op(:matmul, identity_0, tensor.items[1], transpose_b: true,
                                                     pad_zeros: true,
                                                     name:        'matrix_dx')
          matmul_db = op(:matmul, tensor.items[0], identity_1, transpose_a: true,
                                                     pad_zeros: true,
                                                     name:        'matrix_dy')

          # begin_a = op(:zeros, op(:rank, tensor.items[0]), nil, data_type: :int32, name: 'begin_a')
          # begin_b = op(:zeros, op(:rank, tensor.items[1]), nil, data_type: :int32, name: 'begin_b')

          # end_a = op(:shape, tensor.items[0])
          # end_b = op(:shape, tensor.items[1])
          # norm_a = op(:slice, matmul_da, begin_a, size: end_a)
          # norm_b = op(:slice, matmul_db, begin_b, size: end_b)

          zero_vect = op(:zeros, target_shape, nil, name: 'zero_vect')

          norm_a = op(:mul, derivative_a, matmul_da, name: 'grad_a_norm_mul_da')
          norm_b = op(:mul, derivative_b, matmul_db, name: 'grad_b_norm_mul_db')
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
      end.tap do |ops|
        if options[:graph]
          options[:graph].add_node!(gradient_program_name, ops)
        end
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

    def self.grad_with_broadcast(tensor, dx, func, options)
      grad = derivative(tensor.items[0], dx, options)
      grad2 = derivative(tensor.items[1], dx, options)
      elements1 = op(:reduce_prod, op(:shape, tensor.items[0]), data_type: :float32)
      elements2 = op(:reduce_prod, op(:shape, tensor.items[1]), data_type: :float32)
      multiplier = elements1 / elements2
      func.call(grad, grad2 * multiplier)
    end

    def self._include?(arr, obj)
      arr.each { |a| return true if a.equal?(obj) }
      false
    end
  end
end