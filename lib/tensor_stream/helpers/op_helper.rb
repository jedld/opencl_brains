module TensorStream
  module OpHelper
    def op(code, a, b = nil, options = {})
      Operation.new(code.to_sym, a, b, options)
    end

    def cons(value, options = {})
      TensorStream.constant(value, options)
    end
  end
end
