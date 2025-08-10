export async function getChops() {
  const res = await fetch("/api/chops");
  return res.json();
}
