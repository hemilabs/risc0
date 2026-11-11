Vertex Fetching (mtl-fetch-all-unaligned) ***
define void @test_kernel(i32 addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %0, i32 addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %1, <4 x i32> addrspace(2)* nocapture noundef readonly align 16 dereferenceable(16) "air-buffer-no-alias" %2, i32 addrspace(3)* nocapture noundef readnone "air-buffer-no-alias" %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = alloca [4 x i32], align 4
  %8 = alloca %struct.A, align 4
  %9 = load <4 x i32>, <4 x i32> addrspace(2)* %2, align 16, !alias.scope !24, !noalias !31
  %10 = extractelement <4 x i32> %9, i64 0
  %11 = icmp ugt i32 %10, %4
  br i1 %11, label %12, label %68

12:                                               ; preds = %6
  %13 = extractelement <4 x i32> %9, i64 2
  %14 = mul i32 %13, %4
  %15 = zext i32 %14 to i64
  %16 = bitcast [4 x i32]* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #4
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = bitcast %struct.A* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %17) #4
  %18 = getelementptr inbounds %struct.A, %struct.A* %8, i64 0, i32 0, i64 0
  %19 = getelementptr inbounds %struct.A, %struct.A* %8, i64 0, i32 0, i64 4
  br label %20

20:                                               ; preds = %20, %12
  %21 = phi %struct.Elem* [ %18, %12 ], [ %23, %20 ]
  %22 = getelementptr inbounds %struct.Elem, %struct.Elem* %21, i64 0, i32 0
  store i32 100, i32* %22, align 4, !tbaa !38
  %23 = getelementptr inbounds %struct.Elem, %struct.Elem* %21, i64 1
  %24 = icmp eq %struct.Elem* %23, %19
  br i1 %24, label %25, label %20

25:                                               ; preds = %20
  %26 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %15
  br label %31

27:                                               ; preds = %31
  %28 = shl i32 %4, 5
  %29 = zext i32 %28 to i64
  %30 = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 %29
  br label %47

31:                                               ; preds = %31, %25
  %32 = phi i32 [ %38, %31 ], [ 0, %25 ]
  %33 = add i32 %32, %4
  %34 = mul i32 %33, -1640531535
  %35 = add i32 %34, -1640531535
  call fastcc void @_ZL6calleePU9MTLdevicejPU14MTLthreadgroupjPjjj(i32 addrspace(1)* noundef %26, i32 noundef %32, i32 noundef %35) #5, !alias.scope !43, !noalias !44
  %36 = zext i32 %32 to i64
  %37 = getelementptr inbounds %struct.A, %struct.A* %8, i64 0, i32 0, i64 %36, i32 0
  store i32 %35, i32* %37, align 4, !tbaa !38
  %38 = add nuw nsw i32 %32, 1
  %39 = icmp eq i32 %38, 4
  br i1 %39, label %27, label %31, !llvm.loop !45

40:                                               ; preds = %47
  store i32 %59, i32 addrspace(1)* %30, align 4, !tbaa !48, !alias.scope !49, !noalias !50
  %41 = getelementptr inbounds i32, i32 addrspace(1)* %30, i64 1
  store i32 0, i32 addrspace(1)* %41, align 4, !tbaa !48, !alias.scope !49, !noalias !50
  %42 = and i32 %4, 3
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds [4 x i32], [4 x i32]* %7, i64 0, i64 %43
  %45 = load i32, i32* %44, align 4, !tbaa !48
  %46 = icmp eq i32 %45, -559038737
  br i1 %46, label %65, label %67

47:                                               ; preds = %47, %27
  %48 = phi i32 [ 0, %27 ], [ %59, %47 ]
  %49 = phi i32 [ 0, %27 ], [ %63, %47 ]
  %50 = add i32 %49, %4
  %51 = mul i32 %50, -1640531535
  %52 = add i32 %51, -1640531535
  %53 = zext i32 %49 to i64
  %54 = getelementptr inbounds %struct.A, %struct.A* %8, i64 0, i32 0, i64 %53, i32 0
  %55 = load i32, i32* %54, align 4, !tbaa !38
  %56 = icmp eq i32 %55, %52
  %57 = shl nuw nsw i32 1, %49
  %58 = select i1 %56, i32 0, i32 %57
  %59 = or i32 %58, %48
  %60 = add nuw nsw i32 %49, 2
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds i32, i32 addrspace(1)* %30, i64 %61
  store i32 %55, i32 addrspace(1)* %62, align 4, !tbaa !48, !alias.scope !49, !noalias !50
  %63 = add nuw nsw i32 %49, 1
  %64 = icmp eq i32 %63, 4
  br i1 %64, label %40, label %47, !llvm.loop !51

65:                                               ; preds = %40
  %66 = getelementptr inbounds i32, i32 addrspace(1)* %30, i64 31
  store i32 0, i32 addrspace(1)* %66, align 4, !tbaa !48, !alias.scope !49, !noalias !50
  br label %67

67:                                               ; preds = %65, %40
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %17) #4
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #4
  br label %68

68:                                               ; preds = %67, %6
  ret void
}
