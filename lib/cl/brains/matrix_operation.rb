
module Cl
    module Brains
        class MatrixOperation
            def initialize(context, prog, queue, a, b)
                @context = context
                @cols = a[0].size
                @rows = b.size
                @queue = queue
                @prog = prog
                matrix_a = NArray.sfloat(a.size * a[0].size)
                matrix_b = NArray.sfloat(b.size * b[0].size)
                                
                a.flatten.each_with_index do |element, index|
                    matrix_a[index] = element
                  end
          
                b.flatten.each_with_index do |element, index|
                    matrix_b[index] = element
                end
                ap "element size #{matrix_a.element_size}"
                @b_matrix_a = @context.create_buffer(matrix_a.size * matrix_a.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => matrix_a)
                @b_matrix_b = @context.create_buffer(matrix_b.size * matrix_b.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => matrix_b)
            end

            def mul_float
                matrix_c = NArray.sfloat(@cols * @rows)
                b_matrix_c = @context.create_buffer(matrix_c.size * matrix_c.element_size)
                f = OpenCL::Int1::new(@cols)
                g = OpenCL::Int1::new(@rows)
                event = @prog.matrix(@queue, [@cols * @rows], f, g, @b_matrix_a, @b_matrix_b, b_matrix_c)
                @queue.enqueue_read_buffer(b_matrix_c, matrix_c, :event_wait_list => [event])
                @queue.finish
                matrix_c.to_a.each_slice(@cols).collect { |slice| slice }
            end
        end
    end
end