// 基础交互脚本：移动端导航与小型增强功能

function initNavigationToggle() {
  const toggle = document.querySelector(".site-nav__toggle");
  const menu = document.querySelector(".site-nav__list");

  if (!toggle || !menu) return;

  toggle.addEventListener("click", () => {
    const isOpen = menu.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", String(isOpen));
  });

  menu.addEventListener("click", (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && target.matches("a[href^='#']")) {
      menu.classList.remove("is-open");
      toggle.setAttribute("aria-expanded", "false");
    }
  });
}

function initSmoothScroll() {
  const links = document.querySelectorAll("a[href^='#']");

  links.forEach((link) => {
    link.addEventListener("click", (event) => {
      const href = link.getAttribute("href");
      if (!href || href === "#") return;

      const target = document.querySelector(href);
      if (!target) return;

      event.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

function initCurrentYear() {
  const yearSpan = document.getElementById("js-current-year");
  if (!yearSpan) return;
  yearSpan.textContent = String(new Date().getFullYear());
}

document.addEventListener("DOMContentLoaded", () => {
  initNavigationToggle();
  initSmoothScroll();
  initCurrentYear();
});

