
module Cl
    module Brains
        class MatrixOperation
            def initialize(context, prog, queue, cols, a, b)
                @context = context
                @cols = cols
                @queue = queue
                @prog = prog
                matrix_a = NArray.int(cols * cols)
                matrix_b = NArray.int(cols * cols)
                                
                a.flatten.each_with_index do |element, index|
                    matrix_a[index] = element
                  end
          
                b.flatten.each_with_index do |element, index|
                    matrix_b[index] = element
                end

                @b_matrix_a = @context.create_buffer(matrix_a.size * matrix_a.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => matrix_a)
                @b_matrix_b = @context.create_buffer(matrix_b.size * matrix_b.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => matrix_b)
            end

            def mul_int
                matrix_c = NArray.int(@cols * @cols)
                b_matrix_c = @context.create_buffer(matrix_c.size * matrix_c.element_size)
                f = OpenCL::Int1::new(@cols)
                event = @prog.matrix(@queue, [@cols * @cols], f, @b_matrix_a, @b_matrix_b, b_matrix_c)
                @queue.enqueue_read_buffer(b_matrix_c, matrix_c, :event_wait_list => [event])
                @queue.finish
                matrix_c.to_a.each_slice(@cols).collect { |slice| slice }
            end
        end
    end
end