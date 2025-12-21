// Model.jsx
import React, { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, useGLTF } from "@react-three/drei";

function GLBModel({ url, scale = 1, rotationSpeed = 0.003 }) {
  const { scene } = useGLTF(url);
  const ref = useRef();

  useFrame(() => {
    if (ref.current) {
      ref.current.rotation.y += rotationSpeed;
    }
  });

  return <primitive ref={ref} object={scene} scale={scale} />;
}

export default function ModelViewer({
  modelUrl,
  height = 500,
  scale = 1,
  rotationSpeed = 0.003,
}) {
  if (!modelUrl) {
    return null;
  }

  return (
    <div style={{ width: "100%", height }}>
      <Canvas camera={{ position: [0, 1.2, 3], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, 5, 3]} intensity={1.2} />

        <Suspense fallback={null}>
          <GLBModel url={modelUrl} scale={scale} rotationSpeed={rotationSpeed} />
          <Environment preset="city" />
        </Suspense>

        <OrbitControls enableDamping />
      </Canvas>
    </div>
  );
}
