/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Tren Vercel that (bien moi truong VERCEL duoc no tu dat), api/*.py duoc
  // Vercel Python runtime phuc vu truc tiep — KHONG rewrite. Khi chay
  // `npm run dev` cuc bo (khong qua `vercel dev`), proxy /api/* sang
  // dev_server.py (xem README) de test giao dien ma khong can dang nhap
  // Vercel.
  async rewrites() {
    if (process.env.VERCEL) return [];
    return [
      { source: "/api/:path*", destination: "http://127.0.0.1:8787/api/:path*" },
    ];
  },
};

module.exports = nextConfig;
