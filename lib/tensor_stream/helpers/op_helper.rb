module TensorStream
  module OpHelper
    def op(code, a, b = nil, options = {})
      Operation.new(code.to_sym, a, b, options)
    end

    def cons(value, options = {})
      TensorStream.constant(value, options)
    end


    def shape_eval(input)
      return [] unless input.kind_of?(Array)
      arr = []
      arr_ptr = input

      Kernel.loop do
        arr << arr_ptr.size
        arr_ptr = arr_ptr[0]

        break unless arr_ptr.is_a?(Array)
      end

      arr
    end
  end
end
